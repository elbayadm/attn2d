import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .masked_convolution import MaskedConvolution



class TinyResNet(nn.Module):
    """ 
    A set of convolutional layers with cumulative residual skip connections
    Similar to ResNet without argsa - !!unidirectional 
    """
    def __init__(
        self, 
        num_features, 
        bottleneck,
        ffn_dim,
        num_layers=8, 
        kernel_size=3, 
        drop_rate=0.1, 
        div=2, 
        add_conv_relu=False,
        bias=False,
        groups=1,
    ):
        super().__init__()
        if div > 1:
            self.reduce_channels = Linear(num_features, num_features // div)
            num_features = num_features // div
        else:
            self.reduce_channels = None
        self.add_up_scale = 1 / (num_layers + 1)
        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.residual_blocks.append(
                _ResLayer(
                    num_features, 
                    bottleneck, 
                    ffn_dim, 
                    kernel_size,
                    drop_rate, 
                    add_conv_relu, 
                    bias,
                    groups
                ))
            self.output_channels = num_features

    def forward(self, x):
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)
        add_up = self.add_up_scale * x
        for layer in self.residual_blocks:
            x = layer(x)
            add_up += self.add_up_scale * x
        return add_up


class _ResLayer(nn.Module):
    """ Single residual layer label: tinyPA2 """ 
    def __init__(self, 
                 num_features, 
                 bottleneck, 
                 ffn_dim,
                 kernel_size, 
                 drop_rate, 
                 add_conv_relu=False, 
                 bias=False,
                 groups=1,
                ):
        super().__init__()
        self.drop_rate = drop_rate
        self.relu = add_conv_relu
        # choose the padding accordingly:
        padding_trg = (kernel_size - 1) // 2
        padding_src = (kernel_size - 1) // 2
        padding = (padding_trg, padding_src)
        
        self.conv1 = nn.Conv2d(
            num_features,
            bottleneck,
            kernel_size=1,
            stride=1,
            bias=bias,
            groups=1,
        )

        self.mconv2 = MaskedConvolution(
            bottleneck, num_features, kernel_size, 
            padding=padding,
            groups=groups,
            bias=bias,
            unidirectional=True,
        )

        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.scale = 0.5 ** .5

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        if self.relu:
            x = F.relu(x)  # in tinyPA3
        x = self.mconv2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.scale * (x + residual)  # N, C, Tt, Ts
        # FFN:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.scale * (x + residual)
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

