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

from fairseq.modules import (
    MiniMaskedConvolution, LayerNorm,
)


class ConvNetActions(nn.Module):
    """ A network of residual convolutional layers"""

    def __init__(self, num_features, num_layers=8, kernel_size=3):
        super().__init__()
        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.residual_blocks.append(_ResLayer(num_features, kernel_size))
        self.final_ln = LayerNorm(num_features, elementwise_affine=False)
        
    def forward(self, x):
        """
        Input : N, Tt, Ts, C
        Output : N, Tt, Ts, C
        """
        add_up = x
        for layer in self.residual_blocks:
            x = layer(x)
            add_up += x
        return self.final_ln(add_up)


class _ResLayer(nn.Module):
    """ Single residual layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, kernel_size, drop_rate=0.1):
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
            bias=True,
        )

        self.mconv2 = MiniMaskedConvolution(
            num_features, num_features,
            kernel_size, 
            padding=padding,
        )
        self.fc1 = Linear(num_features, 4*num_features)
        self.fc2 = Linear(4*num_features, num_features)
        self.scale = 0.5 ** .5

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
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

