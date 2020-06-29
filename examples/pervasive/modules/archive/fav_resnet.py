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
    MaskedConvolution, MultiheadMaskedConvolution
)


class FavResNet(nn.Module):
    """ A network of residual convolutional layers"""

    def __init__(self, num_init_features, args):
        super().__init__()
        num_layers = args.num_layers
        kernel_size = args.kernel_size
        num_features = num_init_features
        self.reduce_channels = Linear(num_features, num_features // args.divide_channels) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels
        self.output_channels = num_features
        self.add_up_scale = 1 / (num_layers + 1)

        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.residual_blocks.append(_ResLayer(num_features, kernel_size, args))
        
    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        """
        Input : N, Tt, Ts, C
        Output : N, Tt, Ts, C
        """
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)
        add_up = self.add_up_scale * x
        for layer in self.residual_blocks:
            x = layer(x,
                      encoder_mask=encoder_mask,
                      decoder_mask=decoder_mask,
                      incremental_state=incremental_state)
            add_up += self.add_up_scale * x
        return add_up


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
        stride = args.conv_stride  # source dimension stride
        dilsrc = args.source_dilation
        diltrg = args.target_dilation
        resolution = args.maintain_resolution
        if resolution:
            if not stride == 1:
                raise ValueError('Could not maintain the resolution with stride=%d' % stride)

            # choose the padding accordingly:
            padding_trg = diltrg * (kernel_size - 1) // 2
            padding_src = dilsrc * (kernel_size - 1) // 2
            padding = (padding_trg, padding_src)
        else:
            # must maintain the target resolution:
            padding = (diltrg * (kernel_size - 1) // 2, 0)

        # Reduce dim should be dividible by groups
        print('Using groups for the first conv:', args.num_heads)
        self.conv1 = nn.Conv2d(num_features,
                               mid_features,
                               kernel_size=1,
                               stride=1,
                               bias=args.conv_bias,
                               groups=args.num_heads)

        self.mconv2 = MaskedConvolution(
            mid_features, num_features,
            kernel_size, args,
            padding=padding,
        )
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.scale = 0.5 ** .5


    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        residual = x
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.mconv2(x, incremental_state)
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

