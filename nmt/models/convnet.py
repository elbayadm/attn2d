import torch.nn as nn
from .blocks import _Blocks

class ConvNet(nn.Sequential):
    def __init__(self, num_channels, params):
        kernel_size = params['kernel']
        dilation = params['dilation']
        num_sets = params['sets']
        groups = params['groups']
        kernels = params.get('kernels', "").split(',')
        drop_rate = params.get('dropout', .0)
        if len(kernels) > 1:
            kernels = [int(k) for k in kernels]
            print('Setting kernels:', kernels)
            assert len(kernels) == num_sets, \
                    "Number of kernels must match the number of layers"
        else:
            kernels = [kernel_size] * num_sets
        block_type = params['block']
        try:
            block = _Blocks[block_type]
        except:
            raise ValueError('Unknown block type %s' % str(block_type))
        print('Convnet with %d blocks of type %s' % (num_sets, block_type))
        super(ConvNet, self).__init__()
        for s in range(num_sets):
            self.add_module('Block%d' % s,
                            block(num_channels,
                                  num_channels,
                                  kernels[s],
                                  dilation=dilation,
                                  groups=groups,
                                  drop_rate=0,
                                  bias=False))
            if s % 2:
                self.add_module('Drop%d' % s,
                                nn.Dropout(drop_rate)
                                )
        if params['conv_dropout']:
            print('Adding channels dropout')
            self.add_module('dropout', nn.Dropout2d(params['conv_dropout']))

