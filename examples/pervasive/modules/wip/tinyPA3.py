# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.utils.checkpoint as cp

from fairseq.modules import MiniMaskedConvolution


class ResNet3(nn.Module):
    """ A set of convolutional layers with cumulative residual skip connections"""

    def __init__(self, num_features, num_layers=8, kernel_size=3, drop_rate=0.1, args=None):
        super().__init__()
        if args.divide_channels > 1:
            self.reduce_channels = Linear(
                num_features, num_features // args.divide_channels
            )
            num_features = num_features // args.divide_channels
        else:
            self.reduce_channels = None
        self.add_up_scale = 1 / (num_layers + 1)
        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.residual_blocks.append(_ResLayer(num_features, kernel_size, drop_rate))
        self.output_channels = num_features

    def forward(self, x):
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)

        add_up = self.add_up_scale * x
        for layer in self.residual_blocks:
            x = layer(x)
            add_up += self.add_up_scale * x
        return add_up


class _ResLayer3(nn.Module):
    """ Single residual layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, kernel_size, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        dilsrc = 1
        diltrg = 1
        # choose the padding accordingly:
        padding_trg = diltrg * (kernel_size - 1) // 2
        padding_src = dilsrc * (kernel_size - 1) // 2
        padding = (padding_trg, padding_src)
        
        # Reduce dim should be dividible by groups
        self.conv1 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.mconv2 = MiniMaskedConvolution(
            num_features, num_features,
            kernel_size, 
            padding=padding,
            bias=False,
        )
        self.fc1 = Linear(num_features, 4*num_features)
        self.fc2 = Linear(4*num_features, num_features)
        self.scale = 0.5 ** .5

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mconv2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.permute(0, 2, 3, 1)
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



