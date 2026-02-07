from typing import Tuple

from mmcv.runner import force_fp32
from torch import nn
import torch
from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseTransform

__all__ = ["LSSTransform"]


@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        feat=x
        B, N, C, fH, fW = feat.shape# torch.Size([1, 6, 256, 32,88])
        feat = feat.view(B * N, C, fH, fW)  
        
        feat = self.depthnet(feat)#torch.Size([6, 198, 32, 88])，118深度和80图像特征

        depth = feat[:, : self.D].softmax(dim=1)#深度概率，lss,torch.Size([6, 118, 32, 88])

        feat = depth.unsqueeze(1) * feat[:, self.D : (self.D + self.C)].unsqueeze(2)#torch.Size([6, 80, 118, 32, 88])，深度概率已经加权到特征里面了，这里的118（D）就代表之前dbound: Tuple[float, float, float]的离散深度值

        feat = feat.view(B, N, self.C, self.D, fH, fW)
        feat = feat.permute(0, 1, 3, 4, 5, 2)
        return feat

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)#[1,80,256,256]
        x = self.downsample(x)#[1,80,180,180]
        return x
