import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single


class MaskedConv1d(nn.Conv1d):
    """
    Masked (autoregressive) conv1d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, dilation=1,
                 groups=1, bias=False):
        # pad = (dilation * (kernel_size - 1)) // 2
        # print('With dilation=%d, the padding is set to %d' % (dilation, pad))
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           padding=padding,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        self.incremental_state = t.zeros(1, 1, 1)
        _, _, kH = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

    def update(self, x):
        k = self.kernel_size // 2 + 1
        buffer = self.incremental_state
        if buffer.size(-1) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1] = buffer[:, :, 1:].clone()
            buffer[:, :, -1:] = x[:, :, -1:]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding='SAME',
                 dilation=1, groups=1, bias=True):
        padding = _single(self.same_padding(
            kernel_size, dilation)
        ) if padding == 'SAME' else _single(int(padding))
        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size,
            1, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @staticmethod
    def same_padding(kernel_size, dilation):
        width = dilation * kernel_size - dilation + 1
        return width // 2


class _MaskedConv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding='SAME', dilation=1, bias=True):
        implied_kernel_size = kernel_size // 2 + 1
        padding = _single(self.same_padding(
            kernel_size, dilation)
        ) if padding == 'SAME' else _single(int(padding))
        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        self.mask = t.ones(out_channels, in_channels, *kernel_size).byte()
        self.mask[:, :, :implied_kernel_size] = t.zeros(
            out_channels, in_channels, implied_kernel_size
        )

        super(_MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, 1, padding, dilation,
            False, _single(0), 1, bias)

    def forward(self, input):
        return F.conv1d(input, self.masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @property
    def masked_weight(self):
        self.weight.data.masked_fill_(self.mask, 0)
        return self.weight

    @staticmethod
    def same_padding(kernel_size, dilation):
        width = dilation * kernel_size - dilation + 1
        return width // 2

    def cuda(self, device=None):
        super(_MaskedConv1d, self).cuda(device)
        self.mask = self.mask.cuda()
        return self

    def cpu(self):
        super(_MaskedConv1d, self).cpu()
        self.mask = self.mask.cpu()
        return self
