import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TODO Test position encoding cat after modif and check sum
"""


class positional_encoding(nn.Module):
    def __init__(self, num_units, zeros_pad=True, scale=True):
        '''Sinusoidal Positional_Encoding.
        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values
                    of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied
                 by sqrt num_units(check details from paper)
        '''
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        # inputs: A 2d Tensor with shape of (N, T).
        N, T = inputs.size()[0: 2]
        # First part of the PE function: sin and cos argument
        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).long()
        position_enc = torch.Tensor([
            [pos / np.power(10000, 2. * i / self.num_units)
             for i in range(self.num_units)]
            for pos in range(T)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1
        lookup_table = position_enc
        if self.zeros_pad:
            lookup_table = torch.cat((torch.zeros(1, self.num_units),
                                     lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1
        outputs = F.embedding(
            position_ind,
            lookup_table,
            padding_idx,
            None, 2, False, False
        )
        if self.scale:
            outputs = outputs * self.num_units ** 0.5
        return outputs


class ChannelsNormalization(nn.Module):
    def __init__(self, n_channels, eps=1e-3):
        super(ChannelsNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(1, n_channels, 1, 1), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, keepdim=True, dim=1)
        sigma = torch.std(z, keepdim=True, dim=1)
        # print('z:', z.size(), "mu, sigma:", mu.size(), sigma.size())
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out
