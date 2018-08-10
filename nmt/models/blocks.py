import math
import torch.nn as nn
import torch.nn.functional as F
from .conv2d import MaskedConv2d, GatedConv2d
from .modules import LayerNormalization


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(Conv, self).__init__()
        self.conv = MaskedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(ConvRes, self).__init__()
        self.conv = MaskedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class InitScaleGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, drop_rate, **kwargs):
        super(InitScaleGatedBlock, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)
        self.drop_rate = drop_rate
        # init weight
        inconv1 = self.conv1.kernel_size[0] * ((self.conv1.kernel_size[1] - 1)// 2) * in_channels
        std = math.sqrt((4 * (1.0 - drop_rate)) / inconv1)
        self.conv1.weight.data.normal_(mean=0, std=std)
        # self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(mean=0, std=std)
        # self.conv2.bias.data.zero_()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(math.sqrt(.5) * (x + res))
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        if self.drop_rate > 0:
            x_norm = F.dropout(x_norm,
                               p=self.drop_rate,
                               training=self.training)
        return x_norm


class ScaleGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, drop_rate, **kwargs):
        super(ScaleGatedBlock, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)
        self.drop_rate = drop_rate

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(math.sqrt(.5) * (x + res))
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        if self.drop_rate > 0:
            x_norm = F.dropout(x_norm,
                               p=self.drop_rate,
                               training=self.training)
        return x_norm



class InitGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, drop_rate, **kwargs):
        super(InitGatedBlock, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)
        self.drop_rate = drop_rate
        # init weight
        inconv1 = self.conv1.kernel_size[0] * ((self.conv1.kernel_size[1] - 1)// 2) * in_channels
        std = math.sqrt((4 * (1.0 - drop_rate)) / inconv1)
        self.conv1.weight.data.normal_(mean=0, std=std)
        # self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(mean=0, std=std)
        # self.conv2.bias.data.zero_()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(x + res)
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        if self.drop_rate > 0:
            x_norm = F.dropout(x_norm,
                               p=self.drop_rate,
                               training=self.training)
        return x_norm



class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, drop_rate, **kwargs):
        super(GatedBlock, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)
        self.drop_rate = drop_rate

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(x + res)
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        if self.drop_rate > 0:
            x_norm = F.dropout(x_norm,
                               p=self.drop_rate,
                               training=self.training)
        return x_norm


class UpGatedBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(UpGatedBlock3, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.norm = nn.LocalResponseNorm(out_channels//64)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(x + res)
        x_norm = self.norm(x)
        return x_norm


class UpGatedBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(UpGatedBlock2, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(x + res)
        x_norm = self.norm(x)
        return x_norm


class UpGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(UpGatedBlock, self).__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.conv2 = GatedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(x + res)
        x_norm = self.norm(x)
        return x_norm


class GatedConvResNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(GatedConvResNorm, self).__init__()
        self.conv = GatedConv2d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                groups=groups,
                                bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)

    def forward(self, x):
        x = self.relu(x + self.conv(x))
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        return x_norm


class ConvResNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(ConvResNorm, self).__init__()
        self.conv = MaskedConv2d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.relu = nn.ReLU()
        self.norm = LayerNormalization(out_channels)

    def forward(self, x):
        x = self.relu(x + self.conv(x))
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        return x_norm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias)
        self.conv2 = MaskedConv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(x + res)


class ResidualBlockNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, groups, bias, **kwargs):
        super(ResidualBlockNorm, self).__init__()
        self.conv1 = MaskedConv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias)
        self.conv2 = MaskedConv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias)
        self.norm = LayerNormalization(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(x + res)
        N, D, Tt, Ts = x.size()
        x = x.permute(0, 2, 3, 1).view(N, Tt*Ts, -1)
        x_norm = self.norm(x)
        x_norm = x_norm.view(N, Tt, Ts, D).permute(0, 3, 1, 2)
        return x_norm


_Blocks = {
    "conv": ConvResNorm,  # 3
    "residual": ResidualBlock,  # 4
    "gated": GatedConvResNorm,  # 6
    "gated-residual": GatedBlock,  # 7
    "init-gated-residual": InitGatedBlock,  # 7
    "init-scale-gated-residual": InitScaleGatedBlock,  # 7
    "scale-gated-residual": ScaleGatedBlock,  # 7

    "up-gated-residual": UpGatedBlock,
    "up2-gated-residual": UpGatedBlock2,
    "up3-gated-residual": UpGatedBlock3,
}


