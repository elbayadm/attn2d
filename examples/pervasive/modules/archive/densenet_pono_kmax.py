# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from fairseq.modules import (
    MaskedConvolution
)

# FIXME non-zero padding is not effective


class DenseNetPONOKmax(nn.Module):
    """ Single block DenseNet with checkpointing"""

    def __init__(self, num_init_features, args):
        super().__init__()
        divide_channels = args.divide_channels
        num_layers = args.num_layers
        growth_rate = args.growth_rate
        num_features = num_init_features
        self.reduce_channels = Linear(
            num_features,
            num_features // args.divide_channels
        ) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels

        cnn_init_features = num_features
        self.dense_layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.dense_layers.append(_DenseLayer(num_features, args))
            num_features += growth_rate

        self.output = _OutputLayer(cnn_init_features)
        self.output_channels = cnn_init_features 

    def forward(self, x, encoder_mask=None, decoder_mask=None, incremental_state=None):
        """
        Input : B, Tt, Ts, C
        Output : B, Tt, Ts, C
        """
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)
        # B,Tt,Ts,C  >>  B,C,Tt,Ts
        x = x.permute(0, 3, 1, 2)

        features = [x]
        for i, layer in enumerate(self.dense_layers):
            x = layer(features,
                      decoder_mask=decoder_mask,
                      encoder_mask=encoder_mask,
                      incremental_state=incremental_state)
            features.append(x)

        x = torch.cat(features, 1)

        # Back to the original shape B, Tt,Ts,C
        x = x.permute(0, 2, 3, 1)
        x = self.output(x)
        return x


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, args):
        super().__init__()
        self.memory_efficient = args.memory_efficient
        self.drop_rate = args.convolution_dropout
        bn_size = args.bn_size
        growth_rate = args.growth_rate
        inter_features = bn_size * growth_rate
        kernel_size = args.kernel_size

        self.norm1 = PONO(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features,
                               inter_features,
                               kernel_size=1,
                               stride=1,
                               bias=args.conv_bias,
                              )
        self.norm2 = PONO(inter_features)
        dilsrc = args.source_dilation
        diltrg = args.target_dilation
        padding_trg = diltrg * (kernel_size - 1) // 2
        padding_src = dilsrc * (kernel_size - 1) // 2
        padding = (padding_trg, padding_src)

        self.mconv2 = MaskedConvolution(
            inter_features, growth_rate,
            kernel_size, args,
            padding=padding,
        )

    def bottleneck_function(self, *inputs):
        x = self.norm1(torch.cat(inputs, 1))
        x = F.relu(x)
        x = self.conv1(x)
        return x

    def forward(self, prev_features, 
                encoder_mask=None, decoder_mask=None,
                incremental_state=None):
        """
        Memory efficient forward pass with checkpointing
        Each DenseLayer splits its forward into:
            - bottleneck_function 
            - therest_function
        Prev_features as list of features in (B, C, Tt, Ts) 
        Returns the new features alone (B, g, Tt, Ts)
        """
        if self.memory_efficient and any(
            prev_feature.requires_grad 
            for prev_feature in prev_features
        ):
            # Does not keep intermediate values,
            # but recompute them in the backward pass:
            # tradeoff btw memory & compute
            x = cp.checkpoint(
                self.bottleneck_function,
                *prev_features
            )
        else:
            x = self.bottleneck_function(*prev_features)

        x = self.norm2(x)
        x = F.relu(x)
        x = self.mconv2(x, incremental_state)
        if encoder_mask is not None:
            x = x.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(1), 0)
        if decoder_mask is not None:
            x = x.masked_fill(decoder_mask.unsqueeze(1).unsqueeze(-1), 0)

        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class _OutputLayer(nn.Module):
    """ Yield max num_features """
    def __init__(self, num_features):
        super().__init__()
        self.k = num_features

    def forward(self, x):
        x, _ = torch.topk(x, k=self.k, dim=-1)
        return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class PONO(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        """
        x in B,C,Tt,Ts
        """
        mean = x.mean(1, keepdim=True) # B,1,Tt,Ts
        std = x.std(1, keepdim=True)
        x = self.gamma * (x - mean) / (std + self.eps) + self.beta 
        return x

