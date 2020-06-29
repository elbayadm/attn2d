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


class ResNet2(nn.Module):
    """ A network of residual convolutional layers"""

    def __init__(self, num_init_features, args):
        super().__init__()
        num_layers = args.num_layers
        kernel_size = args.kernel_size
        num_features = num_init_features
        self.reduce_channels = Linear(
            num_features,
            num_features // args.divide_channels
        ) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels
        self.output_channels = num_features
        layer_type = args.layer_type

        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            if layer_type == 'macaron':
                _Module = _MacaronResLayer
            else:
                _Module = _ResLayer
            self.residual_blocks.append(_Module(num_features, kernel_size, args))
        
    def forward(self, x,
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None
               ):

        """
        Input : N, Tt, Ts, C
        Output : N, Tt, Ts, C
        """
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)
        for layer in self.residual_blocks:
            x = layer(x, incremental_state)
        return x


class _MacaronResLayer(nn.Module):
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

        self.pre_fc1 = Linear(num_features, ffn_dim)
        self.pre_fc2 = Linear(ffn_dim, num_features)
        self.pre_ffn_ln = nn.LayerNorm(num_features)

        # Reduce dim should be dividible by groups
        self.conv1 = nn.Conv2d(num_features,
                               mid_features,
                               kernel_size=1,
                               stride=1,
                               bias=False)

        self.mconv2 = MaskedConvolution(
            mid_features, num_features,
            kernel_size, args,
            padding=padding,
        )
        self.conv_ln = nn.LayerNorm(num_features)
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.ffn_ln = nn.LayerNorm(num_features)
        self.scale = 2 ** .5

    def forward(self, x,
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None
               ):

        """
        Input x has the shape (N, Tt, Ts, C)
        """
        # First FFN
        residual = x
        x = self.pre_fc1(x)
        x = F.relu(x)
        x = self.pre_fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        # Residual connection
        x = self.scale * (x + residual)
        x = self.pre_ffn_ln(x)

        residual = x
        x = x.permute(0, 3, 1, 2)  # N,Tt,Ts,C >> N,C,Tt,Ts
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mconv2(x, incremental_state)
        if encoder_mask is not None:
            x = x.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(1), 0)
        if decoder_mask is not None:
            x = x.masked_fill(decoder_mask.unsqueeze(1).unsqueeze(-1), 0)

        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.permute(0, 2, 3, 1)  # back to N,Tt,Ts,C
        # Residual connection
        x = self.scale * (x + residual)
        x = self.conv_ln(x)

        # FFN:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        # Residual connection
        x = self.scale * (x + residual)
        x = self.ffn_ln(x)
        return x


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
        self.conv1 = nn.Conv2d(num_features,
                               mid_features,
                               kernel_size=1,
                               stride=1,
                               bias=False)

        self.mconv2 = MaskedConvolution(
            mid_features, num_features,
            kernel_size, args,
            padding=padding,
        )
        self.ln1 = nn.LayerNorm(num_features)
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.ln2 = nn.LayerNorm(num_features)
        self.scale = 2 ** .5

    def forward(self, x, incremental_state):
        residual = x
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mconv2(x, incremental_state)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.permute(0, 2, 3, 1)
        x = self.scale * (x + residual)  # N, C, Tt, Ts
        x = self.ln1(x)
        # FFN:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.scale * (x + residual)
        x = self.ln2(x)
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

