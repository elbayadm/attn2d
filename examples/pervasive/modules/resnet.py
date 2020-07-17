import torch
import torch.nn as nn
import torch.nn.functional as F


from . import register_convnet
from .convnet import ConvNet
from .masked_convolution import MaskedConvolution


@register_convnet("resnet")
class ResNet(ConvNet):
    """ 
    A network of residual convolutional layers
    label : resnet_addup_nonorm2
    """

    def __init__(self, args, num_features):
        super().__init__()
        num_layers = args.num_layers
        kernel_size = args.kernel_size
        self.reduce_channels = Linear(
            num_features, num_features // args.divide_channels
        ) if args.divide_channels > 1 else None
        num_features = num_features // args.divide_channels
        self.output_channels = num_features
        self.add_up_scale = 1 / (num_layers + 1)
        self.nonzero_padding = args.nonzero_padding

        self.residual_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.residual_blocks.append(_ResLayer(num_features, kernel_size, args))
        # for name, p in self.named_parameters():
            # print(name, p.size())
        
    @staticmethod
    def add_args(parser):
        ConvNet.add_args(parser)
        parser.add_argument(
            "--ffn-dim", type=int, default=1024,
            help="FFN dimension"
        )
        parser.add_argument(
            "--bottleneck", type=int, default=256,
            help="bottleneck dimension"
        )
        parser.add_argument(
            "--add-conv-relu", action='store_true', default=False,
            help="Add relu to the depthwise-seprabale convolution"
        )


        
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
        self.relu = args.add_conv_relu
        ffn_dim = args.ffn_dim
        mid_features = args.bottleneck
        stride = args.conv_stride  # source dimension stride
        dilsrc = args.source_dilation
        diltrg = args.target_dilation
        if not args.downsample:
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
                               bias=True, # args.conv_bias,
                               groups=1)  # num-heads??

        self.mconv2 = MaskedConvolution(
            mid_features, num_features, kernel_size, 
            padding=padding,
            stride=args.conv_stride,
            groups=args.conv_groups,
            bias=args.conv_bias,
            unidirectional=args.unidirectional,
            source_dilation=dilsrc,
            target_dilation=diltrg,
        )
        self.fc1 = Linear(num_features, ffn_dim)
        self.fc2 = Linear(ffn_dim, num_features)
        self.scale = 0.5 ** .5
        self.nonzero_padding = args.nonzero_padding


    def forward(self, x, 
                encoder_mask=None,
                decoder_mask=None,
                incremental_state=None):
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        if self.relu: # resnet-addup-nonorm vs resnet-addup-nonorm2 (False)
            x = F.relu(x)
        x = self.mconv2(x, incremental_state)
        # if self.training and not self.nonzero_padding:
        if not self.nonzero_padding:
            if encoder_mask is not None:
                x = x.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(1), 0)
            if decoder_mask is not None:
                x = x.masked_fill(decoder_mask.unsqueeze(1).unsqueeze(-1), 0)

        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.permute(0, 2, 3, 1).contiguous()
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

