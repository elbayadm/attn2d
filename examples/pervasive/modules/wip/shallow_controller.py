# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.utils.checkpoint as cp

from fairseq.modules import (
    LayerNorm,
    SimulTransOracle,
    GridMAX,
)
from fairseq import utils


class ShallowController(nn.Module):

    def __init__(self, args, num_features):
        super().__init__()
        self.args = args
        self.remove_writer_dropout = args.control_remove_writer_dropout

        self.final_ln = LayerNorm(num_features, elementwise_affine=True) \
                if args.control_add_final_ln else None

        if args.control_aggregation == 'max':
            self.aggregator = GridMAX(num_features)
        elif args.control_aggregation == 'cell':
            self.aggregator = lambda x: (x, None)
        else:
            raise ValueError('Unknown aggregation for the controller', args.control_aggregation)
        
        self.net = nn.Sequential()
        for _ in range(args.control_num_layers):
            self.net.add_module('lin%d'%_, Linear(num_features, num_features//2))
            num_features = num_features // 2

        # Oracle:
        self.oracle = SimulTransOracle(
            args.control_oracle_penalty
        ) 

        # Agent : Observation >> Binary R/W decision
        self.gate = nn.Linear(num_features, 1, bias=True)
        nn.init.normal_(self.gate.weight, 0, 1 / num_features)
        nn.init.constant_(self.gate.bias, 0)
        self.write_right = args.control_write_right
        
    def forward(self, sample, encoder_out, decoder_out):
        x = decoder_out[1]
        # Final LN
        if self.final_ln is not None:
            x = self.final_ln(x)
        # Aggregate
        x, _ = self.aggregator(x)
        # A stack of linear layers
        x =  self.net(x)
        # The R/W decisions:
        x = self.gate(x)
        s = F.logsigmoid(x)
        RWlogits = torch.cat((s, s-x), dim=-1).float()

        lprobs = decoder_out[0] 
        target = sample['target']
        encoder_mask = encoder_out['encoder_padding_mask']
        decoder_mask = decoder_out[2]

        with torch.no_grad():
            # Gather the ground truth likelihoods
            B, Tt, Ts, V = lprobs.size()
            lprobs = utils.log_softmax(lprobs, dim=-1)
            scores = lprobs.view(-1, V).gather(
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)  # BTtTs
            ).view(B, Tt, Ts)
            # Forbid padding positions:  # I'm using NLL beware
            if encoder_mask is not None:
                scores = scores.masked_fill(encoder_mask.unsqueeze(1), -1000)
            if decoder_mask is not None:
                scores = scores.masked_fill(decoder_mask.unsqueeze(-1), -1000)

            # The Oracle
            best_context = self.oracle(scores)

            # AP = best_context.add(1).float().mean(dim=1) / Ts
            # print('AP:', ' '.join(map(lambda x: '{:.2f}'.format(x), AP.tolist())))
            Gamma = torch.zeros_like(scores).scatter_(-1, best_context.unsqueeze(-1), 1.0)  # B, Tt, Ts
            
        # Write beyond the ideal context
        if self.write_right:
            Gamma = Gamma.cumsum(dim=-1)
            write = Gamma[:, 1:]  # B, Tt-1, Ts
        else:
            write = Gamma[:, 1:].cumsum(dim=-1)  # B, Tt-1, Ts
        read = 1 - write
        return Gamma, RWlogits[:, :-1], read, write

    def decide(self, x):
        torch.set_printoptions(precision=2)
        # Final LN
        if self.final_ln is not None:
            x = self.final_ln(x)
        # Aggregate
        x, _ = self.aggregator(x)
        x = x[:, -1, -1]
        # A stack of linear layers
        x =  self.net(x)
        # The R/W decisions:
        x = torch.sigmoid(self.gate(x)).squeeze(-1)  # p(read)
        return  1-x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def _expand(tensor, dim, reps):
    tensor = tensor.unsqueeze(dim)
    shape = tuple(reps if i == dim else -1 for i in range(tensor.dim()))
    return tensor.expand(shape)


def PositionalEmbedding(num_embeddings, embedding_dim,
                        padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1,
                                       embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad,
                                          num_embeddings + padding_idx + 1)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


