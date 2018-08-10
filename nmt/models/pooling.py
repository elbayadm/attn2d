import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['truncated_max', "truncated_mean",  'average_code', 'max_code',
           'combined_max_avg', 'pick_first',
           'adaptive_avg_code', 'adaptive_max_code',
           'hierarchical_max_code', 'hierarchical_avg_code',
           'PositionalPooling',
           "MaxAttention", "MaxAttention2"]


def truncated_max(tensor, src_lengths, track=False, *args):
    # input size: N, d, Tt, Ts
    # src_lengths : N,
    # n=1
    # print('tensor:', tensor.size(), 'lengths:', src_lengths)
    Pool = []
    Attention = []
    for n in range(tensor.size(0)):
        X = tensor[n]
        xpool, attn = X[:, :, :src_lengths[n]].max(dim=2)
        if track:
            targets = torch.arange(src_lengths[n])
            align = targets.apply_(lambda k: sum(attn[:, -1] == k))
            align = align/align.sum()
            Attention.append(align.unsqueeze(0))
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    if track:
        return result, torch.cat(Attention, dim=0).cuda()
    return result


def truncated_mean(tensor, src_lengths, *args):
    # input size: N, d, Tt, Ts
    # src_lengths : N,
    # n=1
    # print('tensor:', tensor.size(), 'lengths:', src_lengths)
    Pool = []
    Attention = []
    for n in range(tensor.size(0)):
        X = tensor[n]
        xpool = X[:, :, :src_lengths[n]].mean(dim=2)
        xpool *=  math.sqrt(src_lengths[n])
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    return result


def average_code(tensor, *args):
    return tensor.mean(dim=3)


def pick_first(tensor, src_lengths=None, track=False):
    # input size: N, d, Tt, Ts
    # src_lengths : N, 1
    return tensor[:, :, :, 0]


def max_code(tensor, src_lengths=None, track=False):
    # input size: N, d, Tt, Ts
    # src_lengths : N, 1
    if track:
        batch_size, nchannels, _, max_len = tensor.size()
        # print('nchannels:', nchannels, "source length:", max_len)
        xpool, attn = tensor.max(dim=3)
        targets = torch.arange(max_len).type_as(attn)
        align = []
        activ_distrib = []
        activ = []
        for n in range(batch_size):
            # distribution of the argmax indices
            align.append(np.array([
                torch.sum(attn[n, :, -1] == k, dim=-1).data.item() / nchannels
                for k in targets
            ]))
            # weighted distribution of the argmax indices
            activ_distrib.append(np.array([
                torch.sum((attn[n, :, -1] == k).float() * xpool[n, :, -1], dim=-1).data.item()
                for k in targets
            ]))
            # return the sparse tensor (0 if not pooled, value otherwise)
            activ.append(np.array([
                ((attn[n, :, -1] == k).float() * xpool[n, :, -1]).data.cpu().numpy()
                for k in targets
            ]))

        align = np.array(align)
        activ = np.array(activ)
        activ_distrib = np.array(activ_distrib)
        return xpool, (None, align, activ_distrib, activ)
    else:
        return tensor.max(dim=3)[0]


def combined_max_avg(tensor, *args):
    a_pool = tensor.mean(dim=3)
    m_pool = tensor.max(dim=3)[0]
    return torch.cat([m_pool, a_pool], dim=1)


class hierarchical_max_code(nn.Sequential):
    def __init__(self, kernel_size, dilation=1):
        super(hierarchical_max_code, self).__init__()
        pad = (dilation * (kernel_size - 1)) // 2
        self.add_module('Pool1',
                        nn.MaxPool2d(kernel_size,
                                     padding=(pad, 0),
                                     stride=(1, kernel_size)))
    def forward(self, inputs, *args):
        x = super(hierarchical_max_code, self).forward(inputs)
        return x.mean(dim=3)


class hierarchical_avg_code(nn.Sequential):
    def __init__(self, kernel_size, dilation=1):
        super(hierarchical_avg_code, self).__init__()
        pad = (dilation * (kernel_size - 1)) // 2
        self.add_module('Pool1',
                        nn.AvgPool2d(kernel_size,
                                     padding=(pad, 1), # usually 0, but for short sequences added 1
                                     stride=(1, kernel_size)))
    def forward(self, inputs, *args):
        x = super(hierarchical_avg_code, self).forward(inputs)
        return x.max(dim=3)[0]


def adaptive_avg_code(x, out_channels=8):
    x = F.adaptive_avg_pool2d(x, (x.size(2), out_channels))
    return x.max(dim=3)[0]


def adaptive_max_code(x, out_channels=8):
    x = F.adaptive_max_pool2d(x, (x.size(2), out_channels))
    return x.mean(dim=3)


class PositionalPooling(nn.Module):
    def __init__(self, max_length, emb_size):
        super(PositionalPooling4, self).__init__()
        self.src_embedding = nn.Embedding(max_length, emb_size)
        self.trg_embedding = nn.Embedding(max_length, emb_size)
        self.src_embedding.weight.data.fill_(1)
        self.trg_embedding.weight.data.fill_(1)
        self.src_embedding.bias.data.fill_(0)
        self.trg_embedding.bias.data.fill_(0)

        self.max = max_length

    def forward(self, inputs, *args):
        inputs = inputs.permute(0, 2, 3, 1)
        N, Tt, Ts, d = inputs.size()
        src = self.src_embedding(
            torch.arange(Ts).type(torch.LongTensor).cuda()
        )
        trg = self.trg_embedding(
            torch.arange(Tt).type(torch.LongTensor).cuda()
        )
        # print('src & trg:', src.size(), trg.size())
        kernel = torch.matmul(trg, src.t())
        # print('Kernel:', kernel.size())
        kernel = kernel.unsqueeze(0).unsqueeze(-1)
        result = inputs * kernel.expand_as(inputs)
        result = result.mean(dim=2).permute(0, 2, 1)
        return result


class MaxAttention(nn.Module):
    def __init__(self, params, in_channels):
        super(MaxAttention, self).__init__()
        self.in_channels = in_channels
        self.attend = nn.Linear(in_channels, 1)
        self.dropout = params['attention_dropout']
        self.scale_ctx = params.get('scale_ctx', 1)
        if params['nonlin'] == "tanh":
            self.nonlin = F.tanh
        elif params['nonlin'] == "relu":
            self.nonlin = F.relu
        else:
            self.nonlin = lambda x: x
        if params['first_aggregator'] == "max":
            self.max = max_code
        elif params['first_aggregator'] == "truncated-max":
            self.max = truncated_max
        elif params['first_aggregator'] == "skip":
            self.max = None
        else:
            raise ValueError('Unknown mode for first aggregator ', params['first_aggregator'])


    def forward(self, X, src_lengths, track=False, *args):
        if track:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            # print('alpha:', alphas.size(), alphas)
            # alphas : N, Tt, Ts , 1
            context = alphas.expand_as(Xatt) * Xatt
            # Mean over Ts >>> N, Tt, d
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            # Projection N, Tt, d
            if self.max is not None:
                Xpool, tracking = self.max(X,
                                           src_lengths,
                                           track=True)
                feat = torch.cat((Xpool, context), dim=1)
                return feat, (alphas[0, -1, :, 0].data.cpu().numpy(), *tracking[1:])
            else:
                return context
        else:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            # alphas : N, Tt, Ts , 1
            context = alphas.expand_as(Xatt) * Xatt
            # Mean over Ts >>> N, Tt, d
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            # Projection N, Tt, d
            if self.max is not None:
                Xpool = self.max(X, src_lengths)
                return torch.cat((Xpool, context), dim=1)
            else:
                return context


class MaxAttention2(nn.Module):
    def __init__(self, params, in_channels):
        super(MaxAttention2, self).__init__()
        self.in_channels = in_channels
        self.pre_attend = nn.Linear(in_channels, 32)
        self.attend = nn.Linear(32, 1)
        self.dropout = params['attention_dropout']
        self.scale_ctx = params.get('scale_ctx', 1)
        if params['nonlin'] == "tanh":
            self.nonlin = F.tanh
        elif params['nonlin'] == "relu":
            self.nonlin = F.relu
        else:
            self.nonlin = lambda x: x
        if params['first_aggregator'] == "max":
            self.max = max_code
        elif params['first_aggregator'] == "truncated-max":
            self.max = truncated_max
        elif params['first_aggregator'] == "skip":
            self.max = None
        else:
            raise ValueError('Unknown mode for first aggregator ', params['first_aggregator'])

    def forward(self, X, src_lengths, *args):
        N, d, Tt, Ts = X.size()
        Xatt = X.permute(0, 2, 3, 1)
        alphas = self.nonlin((self.pre_attend(Xatt)))
        alphas = self.attend(alphas)
        alphas = F.softmax(alphas, dim=2)
        # alphas : N, Tt, Ts , 1
        context = alphas.expand_as(Xatt) * Xatt
        # Mean over Ts >>> N, Tt, d
        context = context.mean(dim=2).permute(0, 2, 1)
        if self.scale_ctx:
            context = math.sqrt(Ts) * context
        # Projection N, Tt, d
        if self.max is not None:
            Xpool = self.max(X, src_lengths)
            return torch.cat((Xpool, context), dim=1)
        else:
            return context



