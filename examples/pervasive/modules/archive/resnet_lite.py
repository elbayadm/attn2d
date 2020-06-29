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
    MaskedConvolution
)


class ResNetLite(nn.Module):
    """ A network of residual convolutional layers"""

    def __init__(self, num_init_features, args):
        super().__init__()
        divide_channels = args.divide_channels
        num_layers = args.num_layers
        kernel_size = args.kernel_size
        self.features = nn.Sequential()
        num_features = num_init_features

        if divide_channels > 1:
            print('Reducing the input channels from %d to %d' % (num_features,
                                                                 num_features // divide_channels))
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels

        self.output_channels = num_features

        for i in range(num_layers):
            # block (kernel_size
            resblock = _ResLayer(num_features, kernel_size, args)
            self.features.add_module('resblock%d' % (i + 1), resblock)
        
    def forward(self, x):
        return self.features(x)


class _ResLayer(nn.Module):
    """ Single residual layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, kernel_size, args):
        super().__init__()
        self.drop_rate = args.convolution_dropout
        ffn_dim = args.ffn_dim
        mid_features = args.reduce_dim
        self.conv1 = nn.Conv2d(num_features,
                               mid_features,
                               kernel_size=1, stride=1,
                               bias=False)
        
        self.mconv2 = MaskedConvolution(
            mid_features, num_features,
            kernel_size, bias=False,
            separable=True
        )
        self.ln1 = nn.LayerNorm(num_features)
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.ln2 = nn.LayerNorm(num_features)
        self.scale = 2 ** .5

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mconv2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.scale * (x + residual)  # N, C, Tt, Ts
        x = x.permute(0, 2, 3, 1)  # N, Tt, Ts, C
        x = self.ln1(x)
        # FFN:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.ln2(self.scale * (x + residual))
        x = self.scale * (x + residual)
        # back to the pervious shape:
        x = x.permute(0, 3, 1, 2)  # N, C, Tt, Ts
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

