# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion



import logging
from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@register_criterion('dynamic_ll_loss')
class dynamic_ll_loss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.alpha = args.write_scale
        self.beta = args.control_scale

    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--control-scale', default=1., type=float)
        parser.add_argument('--write-scale', default=1, type=float)

    def compute_writing_loss(self, gamma, lprobs, target):
        """
        Target in B, Tt
        Lprobs logp (y_t | ..) in B, Tt, Ts
        Gamma select positions in B, Tt, Ts
        """
        B, Tt, Ts, V = lprobs.size()
        target = target.unsqueeze(-1)  # B, Tt, 1
        non_pad_mask = target.ne(self.padding_idx)  # B, Tt, 1
        smooth_loss = -lprobs.sum(dim=-1)  # B, Tt, Ts
        # Gather ground truth:
        lprobs = lprobs.contiguous().view(-1, V)
        target = target.repeat(1, 1, Ts) # B,Tt,Ts
        nll_loss = -lprobs.gather(dim=-1, index=target.view(-1,1)).view(B, Tt, Ts)
        # Normalize by #contexts
        mask = gamma.ne(0) * non_pad_mask
        denom = mask.sum(-1, keepdim=True).type_as(lprobs)  # B,Tt,1
        denom += (~non_pad_mask).type_as(denom)
        # print('Denom:', denom)
        # print('Gamma:', gamma)
        # print('NLL terms:', nll_loss)
        nll_loss = (gamma * nll_loss / denom)[mask].sum()
        smooth_loss = (gamma * smooth_loss/ denom)[mask].sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_control_loss(self, controls, read_labels, write_labels):
        # controller
        Ts = controls.size(2)
        rmask = torch.isfinite(controls[..., 0])
        wmask = torch.isfinite(controls[..., 1])
        read_loss =  torch.sum(read_labels[rmask] * controls[...,0].float()[rmask]) 
        write_loss =  torch.sum(write_labels[wmask] * controls[...,1].float()[wmask])
        controlling_loss = - (read_loss + write_loss) / controls.size(-1)
        return controlling_loss

    def forward(self, sample, lprobs, controls, gamma, read_labels, write_labels):
        """
        Lprobs : Tt, B, Tt, V
        Controls: Tt-1, B, Ts, 2
        Gamma: Tt, B, Ts
        Read/Write: Tt-1, B, Ts
        """
        # print('Gamma:', gamma.size())
        # print('Lporbs:', lprobs.size())
        # print('controls:', controls.size())
        # print('Read/write', read_labels.size(), write_labels.size())
        writing_loss, nll_loss = self.compute_writing_loss(gamma, lprobs, sample['target'])
        controlling_loss = self.compute_control_loss(controls, read_labels, write_labels)
        # Total
        loss = self.alpha * writing_loss + self.beta * controlling_loss 
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data.item(),
            'writing_loss': writing_loss.data.item(),
            'nll_loss': nll_loss.data.item(),
            'controlling_loss':  controlling_loss.data.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        writing_loss_sum = sum(log.get('writing_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        controlling_loss_sum = sum(log.get('controlling_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'writing_loss': writing_loss_sum / sample_size / math.log(2),
            'nll_loss':  nll_loss_sum / ntokens / math.log(2),
            'controlling_loss': controlling_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
