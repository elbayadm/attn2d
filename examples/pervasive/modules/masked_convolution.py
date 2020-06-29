import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class MaskedConvolution(nn.Conv2d):
    """ 2d convolution with masked kernel """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding,
                 stride=1,
                 groups=1,
                 bias=False, 
                 unidirectional=False,
                 source_dilation=1, 
                 target_dilation=1):

        stride = (1, stride)  # source dimension stride
        self.dilsrc = source_dilation 
        self.diltrg = target_dilation 
    
        super().__init__(in_channels,
                         out_channels, 
                         kernel_size,
                         stride=stride, 
                         padding=padding,
                         dilation=(self.diltrg, self.dilsrc), 
                         bias=bias,
                         groups=groups)

        self.inc = in_channels
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.pad = padding
        mask = self.build_mask(unidirectional)
        self.register_buffer('mask', mask)
        # print('Mask:', self.mask)

    def build_mask(self, unidirectional=False):
        mask = torch.ones_like(self.weight)
        if self.kernel_size > 1:
            mask[:, :, self.kernel_size // 2 + 1:, :] = 0
            if unidirectional:
                mask[:, :, :, self.kernel_size // 2 + 1:] = 0
        assert(mask.shape == self.weight.shape), \
            "Mask of shape {} must match weights of shape {}" \
            .format(mask.shape, self.weight.shape)
        return mask

    def forward_with_update(self, x):
        self.weight.data *= self.mask
        x = super().forward(x)
        return x

    def forward(self, x, incremental_state=None):
        self.weight.data *= self.mask
        saved_state = None
        if incremental_state is not None:
            # check saved context and append it to the input
            saved_state = self._get_input_buffer(incremental_state)
            if 'activations' in saved_state:
                xprev = saved_state['activations']  # B, C, hist, Ts
                diff = x.size(-1) - xprev.size(-1)
                if diff > 0:
                    pd = xprev.new_zeros((xprev.size(0), xprev.size(1), xprev.size(2), diff))
                    xprev = torch.cat((xprev, pd), dim=-1)
                elif diff < 0:
                    xprev = xprev[...,:diff]
                x = torch.cat((xprev, x), dim=2)
            # cache the input
            hist = min(x.size(1), (self.kernel_size // 2)*self.diltrg)
            self._set_input_buffer(incremental_state,
                                   {'activations': x[:, :, -hist:]})
                        
        x = super().forward(x)
        if saved_state is not None:
            # Return the last token
            x = x[:, :, -1:]
        return x

    def _get_input_buffer(self, incremental_state):
        return self.get_incremental_state(
            incremental_state,
            'conv_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        self.set_incremental_state(
            incremental_state,
            'conv_state',
            buffer,
        )

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

