from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv1d import MaskedConv1d


def make_positions(tensor, padding_idx, left_pad):
    len = tensor.size(1)
    max_pos = padding_idx + 1 + len
    out = torch.arange(padding_idx + 1, max_pos).long().cuda()
    mask = tensor.ne(padding_idx)
    positions = out[:len].expand_as(tensor)
    final = tensor.clone().masked_scatter_(mask, positions[mask])
    if left_pad:
        zero_left = torch.zeros(tensor.size(0), 1).type_as(final)
        final = torch.cat([
            torch.zeros(tensor.size(0), 1).type_as(final),
            final[:, :-1]
        ], dim=1)

    return final


class PosEmbedding(nn.Embedding):
    def __init__(self, max_length, position_dim, pad_left=False):
        super(PosEmbedding, self).__init__(max_length, position_dim, 0)
        self.pad_left = pad_left

    def forward(self, labels):
        positions = make_positions(labels, self.padding_idx, self.pad_left)
        # print('Positions:', positions)
        return super().forward(positions)

    def map(self, inputs):
        return super(PosEmbedding, self).forward(inputs)


class Embedding(nn.Module):
    def __init__(self, params,
                 vocab_size, padding_idx,
                 pad_left=False):

        nn.Module.__init__(self)
        self.dimension = params['input_dim']
        self.encode_length = params['encode_length']
        self.encode_position = params['encode_position']
        self.dropout = params['input_dropout']
        max_length = params['max_length']
        self.init_std = params.get('init_std', .01)
        self.zero_pad = params.get('zero_pad', 0)
        self.padding_idx = padding_idx
        self.label_embedding = nn.Embedding(
            vocab_size,
            self.dimension,
            padding_idx,
            scale_grad_by_freq=False
        )

        if self.encode_position:
            self.pos_embedding = PosEmbedding(max_length,
                                              self.dimension,
                                              pad_left=pad_left)

        if self.encode_length:
            self.dimension += self.encode_length
            # Encode Ts and Tt in the embeddings:
            self.length_embedding = nn.Embedding(max_length,
                                                 self.encode_length)

    def init_weights(self):
        std = self.init_std
        self.label_embedding.weight.data.normal_(0, std)
        # fill padding with zero (default in pytorch if not reinitializing)
        if self.zero_pad:
            self.label_embedding.weight.data[self.padding_idx].fill_(0)
        if self.encode_position:
            self.pos_embedding.weight.data.normal_(0, std)
        if self.encode_length:
            self.length_embedding.weight.data.normal_(0, std)

    def forward(self, data):
        labels = data["labels"]
        emb = self.label_embedding(labels)
        if self.encode_position:
            pos = self.pos_embedding(labels)
            # print('lab emb:', emb, emb.dtype, emb.device)
            # print('pos emb:', pos, pos.dtype, pos.device)
            emb = sqrt(0.5) * (emb + pos)
        if self.encode_length:
            lens = self.length_embedding(data['lengths']).unsqueeze(
                1
            ).repeat(1, emb.size(1), 1)
            emb = torch.cat((emb, lens), dim=2)
        if self.dropout:
            emb = F.dropout(emb,
                            p=self.dropout,
                            training=self.training)
        return emb

    def single_token(self, tok, position, length=None):
        emb = self.label_embedding(tok)
        if self.encode_position:
            position = torch.ones((tok.size(0), 1)).type_as(tok) * position
            # print('position:', position)
            pos = self.pos_embedding.map(position)
            emb += pos
        if self.encode_length:
            lens = self.length_embedding(length).unsqueeze(
                1
            ).repeat(1, emb.size(1), 1)
            emb = torch.cat((emb, lens), dim=2)
        if self.dropout:
            emb = F.dropout(emb,
                            p=self.dropout,
                            training=self.training)
        return emb

    def reset_buffers(self):
        pass


class ConvEmbedding(nn.Module):
    def __init__(self, params,
                 vocab_size, padding_idx,
                 is_target=False):
        nn.Module.__init__(self)
        self.dimension = params['input_dim']
        self.encode_length = params['encode_length']
        self.encode_position = params['encode_position']
        self.dropout = params['input_dropout']
        self.nlayers = params['num_layers']
        kernel = params['kernel']
        self.kernel_size = kernel
        pad = (kernel - 1) // 2
        self.label_embedding = nn.Embedding(
            vocab_size,
            self.dimension,
            padding_idx,
            scale_grad_by_freq=False
        )
        self.conv = nn.Sequential()
        if is_target:
            conv = MaskedConv1d
            self.incremental_state = None
        else:
            conv = nn.Conv1d

        for l in range(self.nlayers):
            self.conv.add_module("conv%d" % l,
                                 conv(self.dimension,
                                      self.dimension,
                                      kernel,
                                      padding=pad,
                                      bias=False))

    def init_weights(self):
        std = .01
        self.label_embedding.weight.data.normal_(0, std)

    def forward(self, data):
        labels = data["labels"]
        emb = self.label_embedding(labels)
        emb = emb.permute(0, 2, 1)
        emb = self.conv(emb)
        if self.dropout:
            emb = F.dropout(emb,
                            p=self.dropout,
                            training=self.training)
        emb = emb.permute(0, 2, 1)
        return emb

    def single_token(self, labels, position=0):  # FIXME change to update
        if self.incremental_state is not None:
            if self.incremental_state.size(1) >= self.kernel_size:
                buffer = self.incremental_state
                # shift the buffer and add the recent input:
                buffer[:, :-1] = buffer[:, 1:].clone()
                buffer[:, -1:] = labels[:, -1:]
                labels = buffer
            else:
                buffer = self.incremental_state
                # shift the buffer and add the recent input:
                buffer = torch.cat((buffer, labels), dim=1)
                labels = buffer

        self.incremental_state = labels
        emb = self.label_embedding(labels)
        emb = emb.permute(0, 2, 1)
        for cvl in list(self.conv.children()):
            emb = cvl(emb)
        emb = emb.permute(0, 2, 1)
        return emb

    def reset_buffers(self):
        self.incremental_state = None
        for clv in list(self.conv.children()):
            clv.incremental_state = torch.zeros(1, 1, 1)


