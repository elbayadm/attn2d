import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ 
    Parent class for all convnets (resnet, densenet ..)
    """
    def __init__(self):
        super().__init__()
                
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--kernel-size", type=int, default=11,
            help="Convolution kernel"
        )
        parser.add_argument(
            "--num-layers", type=int, default=14,
            help="Number of layers"
        )
        parser.add_argument(
            "--divide-channels", type=int, default=2,
            help="Factor by which we reduce the input channels (emded_source + embded_target)"
        )
        parser.add_argument(
            "--convolution-dropout", type=float, default=0.2,
            help="dropout rate in the residual layer"
        )

        parser.add_argument(
            "--conv-bias", action='store_true',  
            help="Add bias to the depthwise convolution (k=1)"
        )

        parser.add_argument(
            "--conv-groups", type=int, help="Grouping of 2d convolutions",
        )

        parser.add_argument(
            "--downsample", action='store_true',  
            help="Do not maintain the grid resolutions along the source axis"
        )

        parser.add_argument(
            '--nonzero-padding', 
            action='store_true', 
            help='Do not zero out padding positions in the conv activations'
        )

        parser.add_argument(
            '--memory-efficient',  #FIXME
            action='store_true', 
            help='Use gradient checkpointing'
        )

        parser.add_argument(
            "--unidirectional", action='store_true',  
            help="Mask the future source as well, making the encoding unidirectional"
        )

        parser.add_argument("--conv-stride", type=int, default=1)
        parser.add_argument("--source-dilation", type=int, default=1)
        parser.add_argument("--target-dilation", type=int, default=1)


    def forward(self, x, encoder_mask=None, decoder_mask=None, incremental_state=None):
        """
        Input : N, Tt, Ts, C
        Output : N, Tt, Ts, C
        """
        raise NotImplementedError
        

