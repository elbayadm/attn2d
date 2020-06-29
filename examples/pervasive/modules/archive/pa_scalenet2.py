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


class PAScaleNet2(nn.Module):
    """ A network of convolutional layers"""

    def __init__(self, num_init_features, args):
        super().__init__()
        num_layers = args.num_layers
        num_features = num_init_features
        self.reduce_channels = Linear(num_features, num_features // args.divide_channels) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels
        self.output_channels = num_features
        self.gate_channels = args.gate_channels
        self.gates = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.gate_embeddings = _ScaleLayer(num_features)
        self.depth_gate_1 = _ScaleLayer(num_features)
        self.depth_gate_2 = _ScaleLayer(num_features)

        for _ in range(num_layers):
            self.blocks.append(_ConvLayer(num_features, args))
            self.blocks.append(_FFLayer(num_features, args))
            self.gates.append(_ScaleLayer(num_features))
            self.gates.append(_ScaleLayer(num_features))

        
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

        features = self.gate_embeddings(x)
        for layer, gate in zip(self.blocks, self.gates):
            xlayer = layer(x,
                           encoder_mask=encoder_mask,
                           decoder_mask=decoder_mask,
                           incremental_state=incremental_state)
            features += gate(xlayer)
            x = self.depth_gate_1(x) + self.depth_gate_2(xlayer)
        return features


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class _ScaleLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor(1).fill_(0.3))

    def forward(self, x):
        return self.scale * x


class _ConvLayer(nn.Module):
    """ Single Conv layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, args):
        super().__init__()
        kernel_size = args.kernel_size
        self.zero_out = args.zero_out_conv_input
        self.drop_rate = args.convolution_dropout
        ffn_dim = args.ffn_dim
        mid_features = args.reduce_dim
        stride = args.conv_stride  
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
                               bias=args.conv_bias)

        self.mconv2 = MaskedConvolution(
            mid_features, num_features,
            kernel_size, args,
            padding=padding,
        )

    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        x = x.permute(0, 3, 1, 2)
        # Zero out the conv input
        if self.zero_out and self.training:
            if encoder_mask is not None:
                x = x.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(1), 0)
            if decoder_mask is not None:
                x = x.masked_fill(decoder_mask.unsqueeze(1).unsqueeze(-1), 0)

        # Depthwise separable convolution
        x = self.conv1(x)
        x = self.mconv2(x, incremental_state)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = x.permute(0, 2, 3, 1)
        return x


class _FFLayer(nn.Module):
    """ Single Conv layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, args):
        super().__init__()
        self.drop_rate = args.convolution_dropout
        ffn_dim = args.ffn_dim
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)

    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


