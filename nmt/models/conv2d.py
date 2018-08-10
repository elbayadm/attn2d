import torch
import torch.nn as nn
import time
"""
CHECK OUT:
    init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
    OpenNMT-py/onmt/modules/WeightNorm.py

"""

class MaskedConv2d_ZeroPad(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    with 'manual' padding
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        # print('With dilation=%d, the padding is set to %d' % (dilation, pad))
        super(MaskedConv2d_ZeroPad, self).__init__(in_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   padding=pad,
                                                   groups=groups,
                                                   dilation=dilation,
                                                   bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, mask=None):
        """
        mask: A mask of size N, 1, Tt x Ts
              0 if position beyond both sequences's lengths
        """
        self.weight.data *= self.mask
        x = super(MaskedConv2d_ZeroPad, self).forward(x)
        if mask is not None:
            # print('Total zeroed values:', sum(mask.view(-1)).data.item())
            x = x.masked_fill_(mask.expand_as(x), 0)
        # print('conv output:', x.size())
        return x

    def update(self, x):
        # FIXME in the case of an or mask, it should alsp be used in the evaluation
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class MaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        # print('With dilation=%d, the padding is set to %d' % (dilation, pad))
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           padding=pad,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class GatedConv2d(MaskedConv2d):
    """
    Gated version of the masked conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 bias=False, groups=1):
        super(GatedConv2d, self).__init__(in_channels,
                                          2*out_channels,
                                          kernel_size,
                                          dilation=dilation,
                                          bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super(GatedConv2d, self).forward(x)
        mask, out = x.chunk(2, dim=1)
        mask = self.sigmoid(mask)
        return out * mask


class SwapConv2d(nn.Conv2d):
    """
    Wrap permuting and conv in a module
    """
    def __init__(self, in_channels, out_channels):
        super(SwapConv2d, self).__init__(in_channels, out_channels, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = super(SwapConv2d, self).forward(x)
        return x.permute(0, 2, 3, 1)


class MaskedConv2d_v2(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    Only difference in mask construction.
    No improvement with small kernels.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        # print('With dilation=%d, the padding is set to %d' % (dilation, pad))
        super(MaskedConv2d_v2, self).__init__(in_channels, out_channels,
                                              kernel_size,
                                              padding=pad,
                                              groups=groups,
                                              dilation=dilation,
                                              bias=bias)
        # mask v2
        _, _, kH, kW = self.weight.size()
        self.register_buffer("mask", torch.ones(out_channels,
                                                in_channels,
                                                kH, kW).byte())
        effective_ks = kH // 2 + 1
        self.mask[:, :, :effective_ks, :] = torch.zeros(
            out_channels, in_channels, effective_ks, kW
        )
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x):
        self.weight.data.masked_fill_(self.mask, 0)
        return super(MaskedConv2d_v2, self).forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


