# CBAM：通道注意力 + 空间注意力（Woo et al., ECCV 2018）
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return self.sig(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        cat = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return self.sig(self.conv(cat))


class CBAM(nn.Module):
    """yaml 用法（n 尺度专用，ch 填缩放后的实际通道数）: [-1, 1, CBAM, [64]]"""

    def __init__(self, ch: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(kernel)

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)
