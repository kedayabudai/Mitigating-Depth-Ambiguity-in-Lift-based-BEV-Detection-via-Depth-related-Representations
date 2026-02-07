import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from ..builder import NECKS



def swish(x):
    return x * x.sigmoid()

class LayerCombineModule(nn.Module):
    def __init__(self, num_input=2):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(num_input, dtype=torch.float32).view(1, 1, 1, 1, -1),
            requires_grad=True
        )

    def forward(self, inputs):

        weights = self.weights.relu()
        norm_weights = weights / (weights.sum() + 0.0001)

        out = (norm_weights*torch.stack(inputs, dim=-1)).sum(dim=-1)
        return swish(out)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x 

class SingleBiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, no_norm_on_lateral=True, conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super().__init__()

        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.lateral_combine = nn.ModuleList()
        self.lateral_combine_conv = nn.ModuleList()
        self.out_combine = nn.ModuleList()
        self.out_combine_conv = nn.ModuleList()

        for i, in_channel in enumerate(in_channels):
            if in_channel != out_channels:
                self.lateral_convs.append(ConvModule(
                    in_channel,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False))
            else:
                self.lateral_convs.append(Identity())
            if i != len(in_channels)-1:
                self.lateral_combine.append(LayerCombineModule(2))
                self.lateral_combine_conv.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False)
                )
            if i != 0:
                self.out_combine.append(LayerCombineModule(
                    3 if i != len(in_channels)-1 else 2))
                self.out_combine_conv.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False))

    def forward(self, inputs):

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]#[[1,256,64,64],[1,256,32,32],[1,256,16,16]]
        laterals = laterals + \
            inputs[len(self.lateral_convs):]  # p3,p4,p5,p6,p7

        # top to mid
        outs = [laterals[i] for i in range(len(laterals))]#[[1, 256, 64, 64],[1, 256, 32, 32],[1, 256, 16, 16]]


        if 'scale_factor' in self.upsample_cfg:
            up_feat = F.interpolate(outs[2],**self.upsample_cfg)
        else:
            prev_shape = outs[1].shape[2:]#依次为[32，32]
            up_feat = F.interpolate(
                outs[2], size=prev_shape, **self.upsample_cfg)#[1,256,32,32],16->32
        # weight combine
        outs[1] = self.lateral_combine_conv[1](self.lateral_combine[1]([outs[1], up_feat]))#依次对应outs[1]<->32

        # down to mid
        # print(laterals[i].size())
        down_feat = F.max_pool2d(outs[0], 3, stride=2, padding=1)
        # print(down_feat.size())
        cur_outs = outs[1]
        if 0 != len(laterals)-2:
            cur_inputs = laterals[1]
            outs[1] = self.out_combine[0]([down_feat, cur_outs, cur_inputs])
        else:
            outs[1] = self.out_combine[0]([down_feat, cur_outs])
        outs[1] = self.out_combine_conv[0](outs[1])

        return outs


@NECKS.register_module()
class BiFPN2(nn.Module):#不再像BIFPN那样top to down 和 down to top。而是top to mid and down to mid
    def __init__(self,
                 in_channels,
                 out_channels=160,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 num_repeat=6,
                 scale=None,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(BiFPN2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_repeat = num_repeat
        self.scale = scale
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # if scale is not None:

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.downsample_convs = nn.ModuleList()
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_conv = nn.Sequential(
                    ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False),
                    nn.MaxPool2d(3,2,1)
                    )
                self.downsample_convs.append(extra_conv)

        out_channels = out_channels if self.add_extra_convs else self.in_channels[
            self.backbone_end_level-1]
        self.bi_fpn = nn.ModuleList()
        for i in range(self.num_repeat):
            if i == 0:
                in_channels = self.in_channels[self.start_level:self.backbone_end_level]+[
                    out_channels]*extra_levels
            else:
                in_channels = [self.out_channels]*num_outs
            self.bi_fpn.append(SingleBiFPN(in_channels, self.out_channels, no_norm_on_lateral=no_norm_on_lateral,
                                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_cfg=upsample_cfg))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        outs = list(inputs[self.start_level:self.backbone_end_level])
        used_backbone_levels = len(outs)
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 3, stride=2, padding=1))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(self.num_outs-used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.downsample_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.downsample_convs[i](outs[-1]))

        # p2,p3,p4,p5,p6,p7
        # forward to bifpn
        for i in range(self.num_repeat):
            outs = self.bi_fpn[i](outs)

        if self.scale is None:
            return outs
        else:
            need_out_ids=len(outs)//2
            out_shape=outs[need_out_ids].shape[2:]
            goal_shape=[x * (2**self.scale) for x in out_shape]
            up_feat = F.interpolate(
                    outs[need_out_ids], size=goal_shape, **self.upsample_cfg)
            return up_feat
