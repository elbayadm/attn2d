# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import torch
import torch.nn as nn

from fairseq.modules import (
    MaskedConvolution
)


class DenseNet(nn.Module):
    """ A network of DenseBlocks in sequence """

    def __init__(self, num_init_features, args):
        super().__init__()
        block_layers = args.num_layers
        kernel_size = args.kernel_size
        bn_size = args.bn_size
        growth_rate = args.growth_rate
        drop_rate = args.dropout
        init_weights = args.init_weights
        divide_channels = args.divide_channels
        num_blocks = args.num_blocks

        self.features = nn.Sequential()
        num_features = num_init_features

        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == 'manual':
                std = math.sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels

        for i in range(num_blocks):
            block = _DenseBlock(block_layers, num_features,
                                kernel_size, bn_size,
                                growth_rate, drop_rate, args)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + block_layers * growth_rate
            if not i == num_blocks - 1 or not args.skip_last_trans:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.output_channels = num_features

        self.features.add_module('final_norm', nn.BatchNorm2d(num_features))
        self.features.add_module('final_relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x)


class _DenseBlock(nn.Sequential):
    """ A single connected block of DenseLayers """
    def __init__(self, num_layers, num_input_features, kernel_size,
                 bn_size, growth_rate, drop_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                kernel_size, growth_rate,
                                bn_size, drop_rate, args)
            self.add_module('denselayer%d' % (i+1), layer)


class _DenseLayer(nn.Module):
    """ Single block of DenseNet

    num_input_features - number of input channels to the layer
    kernel_size - size of masked convolution, k x (k // 2)
    growth_rate - number of output channels concatenated to feature map
    bn_size - bottleneck size
    drop_rate - dropout rate
    """

    def __init__(self, num_input_features, kernel_size,
                 growth_rate, bn_size, drop_rate, args, bias=False):
        super().__init__()

        self.drop_rate = drop_rate
        inter_features = bn_size * growth_rate

        self.layer = nn.Sequential()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, inter_features,
                               kernel_size=1, stride=1, bias=bias)

        self.layer2 = nn.Sequential()
        self.layer2.add_module('bn2', nn.BatchNorm2d(inter_features))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))

        mask = torch.ones(growth_rate, inter_features,
                          kernel_size, kernel_size)
        if kernel_size > 1:
            mask[:, :, kernel_size // 2 + 1:, :] = 0
        padding = (kernel_size - 1) // 2

        self.layer2.add_module(
            'mconv2',
            MaskedConvolution(
                mask, inter_features, growth_rate,
                kernel_size, padding, bias=bias
            )
        )
        self.layer2.add_module('do1', nn.Dropout(p=self.drop_rate))

    def forward(self, x):
        # self.first_out = x
        # self.bn_out = self.bn1(self.first_out)
        # self.relu_out = self.relu1(self.bn_out)
        # self.inter_out = self.conv1(self.relu_out)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.layer2(out)
        return torch.cat([x, out], 1)


def Transition(num_input_features, num_output_features):
    """ Return a transition module between dense blocks """
    trans = nn.Sequential(
                nn.BatchNorm2d(num_input_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_input_features,
                          num_output_features,
                          kernel_size=1,
                          stride=1,
                          bias=False)
                )
    return trans


