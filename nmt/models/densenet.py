from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d import MaskedConv2d, GatedConv2d, MaskedConv2d_ZeroPad, MaskedConv2d_v2
_RESET = (MaskedConv2d, MaskedConv2d_v2, MaskedConv2d_ZeroPad)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate,
                 kernel_size=3, bn_size=4, drop_rate=0,
                 gated=False, bias=False,
                 init_weights=True,
                 weight_norm=False,
                 norm='batch',
                 version=0):
        super(_DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        if gated:
            CV = GatedConv2d
        else:
            CV = MaskedConv2d
        conv1 = nn.Conv2d(num_input_features,
                          bn_size * growth_rate,
                          kernel_size=1,
                          bias=bias)
        conv2 = CV(bn_size * growth_rate,
                   growth_rate,
                   kernel_size=kernel_size,
                   bias=bias)

        if init_weights:
            std1 = sqrt(2/num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(2 * (1 - drop_rate) / (bn_size * growth_rate *
                                               kernel_size *
                                               (kernel_size - 1)//2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()

        if weight_norm:
            print('Adding weight normalization')
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)

        # Add all modules
        if norm == "batch":
            self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        # elif norm == "layer":
            # self.add_module('norm1', ChannelsNormalization(num_input_features))
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', conv1)
        if norm == "batch":
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        # elif norm == "layer":
            # self.add_module('norm2', ChannelsNormalization(bn_size * growth_rate))
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', conv2)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # print('new_features:', new_features.size())
        # print('std:', torch.mean(new_features.std(dim=1).view(-1)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.children()):
            if isinstance(layer, MaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.children()):
            if isinstance(layer, MaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)


    def track(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return x, new_features


class _DenseLayer_mask(nn.Module):
    def __init__(self, num_input_features, growth_rate,
                 kernel_size=3, bn_size=4, drop_rate=0,
                 gated=False, bias=False,
                 init_weights=True,
                 weight_norm=False,
                 norm='batch',
                 version=1):
        super(_DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        if gated:
            CV = GatedConv2d
        else:
            if version == 2:
                CV = MaskedConv2d_ZeroPad
            else:
                CV = MaskedConv2d
        conv1 = nn.Conv2d(num_input_features,
                          bn_size * growth_rate,
                          kernel_size=1,
                          bias=bias)
        conv2 = CV(bn_size * growth_rate,
                   growth_rate,
                   kernel_size=kernel_size,
                   bias=bias)

        if init_weights:
            std1 = sqrt(2/num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(2 * (1 - drop_rate) / (bn_size * growth_rate *
                                               kernel_size *
                                               (kernel_size - 1)//2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()

        if weight_norm:
            print('Adding weight normalization')
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)

        # Modules (moving from Sequential)
        self.norm1 = nn.BatchNorm2d(num_input_features)  # then relu
        self.conv1 = conv1
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)  # then relu
        self.conv2 = conv2
        self.drop_rate = drop_rate

    def forward(self, x, mask=None):
        res = x
        x = F.relu(self.norm1(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.norm2(x), inplace=True)
        x = self.conv2(x, mask)
        if self.drop_rate > 0:
            x = F.dropout(x,
                          p=self.drop_rate,
                          training=self.training)
        return torch.cat([res, x], 1)

    def reset_buffers(self):
        """
        Reset all buffers i.e each conv's previous inputs
        Call once sequence sampling is done
        """
        for layer in list(self.children()):
            if isinstance(layer, _RESET):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        x = F.relu(self.norm1(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.norm2(x), inplace=True)
        x = self.conv2.update(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        res = x
        x = F.relu(self.norm1(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.norm2(x), inplace=True)
        x = self.conv2(x)
        return res, x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, kernel_size,
                 bn_size, growth_rate, drop_rate, gated,
                 bias, init_weights, weight_norm, norm, version):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            print('Layer %d: in_channels = %d' % (i, num_input_features +
                                                  i * growth_rate))
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate,
                                kernel_size,
                                bn_size, drop_rate,
                                gated=gated,
                                bias=bias,
                                init_weights=init_weights,
                                weight_norm=weight_norm,
                                norm=norm,
                                version=version)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward_mask(self, x, mask=None):
        x = x.contiguous()
        for layer in list(self.children()):
            x = layer(x, mask)
        return x

    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()

    def track(self, x):
        activations = []
        for layer in list(self.children()):
            # layer is a DenseLayer
            x, newf = layer.track(x)
            activations.append(newf.data.cpu().numpy())
            x = torch.cat([x, newf], 1)
        return x, activations


class _STransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 bias, init_weights, weight_norm):
        super(_STransition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        conv = nn.Conv2d(num_input_features,
                         num_output_features,
                         kernel_size=1,
                         bias=bias)

        if init_weights:
            std = sqrt(2/num_input_features)
            conv.weight.data.normal_(0, std)
            if bias:
                conv.bias.data.zero_()
        if weight_norm:
            conv = nn.utils.weight_norm(conv, dim=0)
        self.add_module('conv', conv)

    def forward(self, x, *args):
        return super(_STransition, self).forward(x)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1,
                                          bias=False)
                        )
        # std = sqrt(2/num_input_features)
        # self.conv.weight.data.normal_(0, std)
        # self.conv.bias.data.zero_()
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x, *args):
        return super(_Transition, self).forward(x)


class DenseNet(nn.Module):
    def __init__(self, num_init_features, params):
        super(DenseNet, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        block_config = params.get('num_layers', (6, 12, 24, 16))
        kernel_size = params.get('kernel', 3)
        bn_size = params.get('bn_size', 4)
        drop_rate = params.get('conv_dropout', 0)
        gated = params.get('gated', 0)
        bias = bool(params.get('bias', 1))
        init_weights = params.get('init_weights', 1)
        weight_norm = params.get('weight_norm', 0)
        norm = params.get('norm', "batch")
        half_init = params.get('half_inputs', 0)
        use_st = params.get('use_st', 1)
        version = params.get('conv_version', 1)

        self.features = nn.Sequential()
        num_features = num_init_features
        # start by halving the input channels
        if half_init:
            trans1 = nn.Conv2d(num_features, num_features//2, 1)
            self.features.add_module('initial_transiton', trans1)
            num_features = num_features // 2

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                kernel_size=kernel_size,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                gated=gated,
                                bias=bias,
                                init_weights=init_weights,
                                weight_norm=weight_norm,
                                norm=norm,
                                version=version)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if use_st:
                trans2 = _STransition(num_input_features=num_features,
                                      num_output_features=num_features // 2,
                                      bias=bias,
                                      init_weights=init_weights,
                                      weight_norm=weight_norm)
            else:
                print('Adding the usual trans')
                trans2 = _Transition(num_input_features=num_features,
                                     num_output_features=num_features // 2
                                     )

            self.features.add_module('transition%d' % (i + 1), trans2)
            num_features = num_features // 2

        self.output_channels = num_features
        # Final batch norm
        self.features.add_module('norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward_mask(self, x, mask=None):
        x = x.contiguous()
        # print('DenseNet children:',
              # [type(c) for c in list(self.features.children())])
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x
        # return self.features(x.contiguous(), mask)

    def forward(self, x):
        return self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                layer.reset_buffers()

    def track(self, x):
        activations = []
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x, actv = layer.track(x)
                activations.append(actv)
            else:
                x = layer(x)
        return x, activations



