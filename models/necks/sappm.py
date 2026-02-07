import torch
import torch.nn as nn
import torch.nn.functional as F

class SAPPM(nn.Module):
    """
    Semantic-Aware Pyramid Pooling Module (SAPPM)
    输入: F_ml  -> B x C x H x W
    输出: F_sappm -> B x C x H x W
    """
    def __init__(self, in_channels, out_channels, pool_sizes=[1,2,4,8]):
        super(SAPPM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes

        # 每个池化分支先降通道再上采样
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Identity 分支
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 最终融合卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels*(len(pool_sizes)+1), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: B x C x H x W
        """
        H, W = x.size(2), x.size(3)
        out = [self.identity(x)]  # Identity branch
        for stage in self.stages:
            y = stage(x)  # B x out_channels x pooled_H x pooled_W
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            out.append(y)
        out = torch.cat(out, dim=1)  # B x (out_channels*(len(pool_sizes)+1)) x H x W
        out = self.fuse(out)  # B x out_channels x H x W
        return out