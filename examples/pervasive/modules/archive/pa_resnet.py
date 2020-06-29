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


class PAResNet(nn.Module):
    """ A network of residual convolutional layers"""

    def __init__(self, num_init_features, args):
        super().__init__()
        num_layers = args.num_layers
        kernel_size = args.kernel_size
        num_features = num_init_features
        self.reduce_channels = Linear(num_features, num_features // args.divide_channels) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels
        self.output_channels = num_features
        self.addup = args.addup
        self.addup_scale = args.addup_scale
        if self.addup in ['ffn', 'attnffn']:
            # If the scale is not given, default to:
            if not self.addup_scale:
                if self.addup == 'ffn':
                    self.addup_scale = 1 / (num_layers + 1)
                elif self.addup == 'attnffn':
                    self.addup_scale = 1 / (2 * num_layers + 1)
        else:
            self.addup_scale = 0

        print('Using addup_scale:', self.addup_scale)
        self.gate_channels = args.gate_channels
        self.gates_ffn = nn.ModuleList([])
        self.gates_attn = nn.ModuleList([])

        self.residual_blocks = nn.ModuleList([])

        for _ in range(num_layers):
            self.residual_blocks.append(_ResLayer(num_features, kernel_size, args))
            if self.gate_channels:
                if 'attn' in self.addup:
                    self.gates_attn.append(_GateLayer(num_features))
                else:
                    self.gates_attn.append(nn.Sequential())

                if 'ffn' in self.addup:
                    self.gates_ffn.append(_GateLayer(num_features))
                else:
                    self.gates_ffn.append(nn.Sequential())
            else:
                self.gates_ffn.append(nn.Sequential())
                self.gates_attn.append(nn.Sequential())

        
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
        if self.addup_scale:
            cumul = self.addup_scale * x
        for layer, gate_ffn, gate_attn in zip(self.residual_blocks, self.gates_ffn, self.gates_attn):
            x, xattn = layer(x,
                             encoder_mask=encoder_mask,
                             decoder_mask=decoder_mask,
                             incremental_state=incremental_state)
            if self.addup_scale:
                if 'ffn' in self.addup:
                    cumul += self.addup_scale * gate_ffn(x)
                if 'attn' in self.addup:
                    cumul += self.addup_scale * gate_attn(xattn)
        if self.addup_scale:
            return cumul
        return x


class _GateLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = Linear(num_features, 2*num_features)

    def forward(self, x):
        return F.glu(self.linear(x), dim=-1)


class _ResLayer(nn.Module):
    """ Single residual layer

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    drop_rate - dropout rate
    """

    def __init__(self, num_features, kernel_size, args):
        super().__init__()
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
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.scale = args.residual_scale 

    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        residual = x
        x = x.permute(0, 3, 1, 2)
        # Zero out the conv input
        if self.training and self.zero_out:
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
        x = self.scale * (x + residual)  # N, C, Tt, Ts
        # FFN:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.scale * (x + residual)
        return x, residual


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

