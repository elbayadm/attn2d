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


@register_criterion('hmm_loss')
class hmm_loss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.control_scale
        self.beta = args.regul_scale
        self.eps = 1e-6
        self.discretize = args.discretize

    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--control-scale', default=1., type=float)
        parser.add_argument('--regul-scale', default=0.1, type=float)
        parser.add_argument('--discretize', action='store_true')

    def compute_writing_loss(self, gamma, emissions):
        mask = gamma.ne(0)
        writing_loss = - (gamma * emissions)[mask].sum()
        return writing_loss

    def compute_control_loss(self, controls, read_labels, write_labels):
        # controller
        print('Writing labels:', write_labels[:,0].data)
        print('Read labels:', read_labels[:,0].data)
        if self.discretize:
            write_labels = torch.gt(write_labels, read_labels).float()
            read_labels = 1 - write_labels
        rmask = torch.isfinite(controls[..., 0])
        wmask = torch.isfinite(controls[..., 1])
        read_loss =  torch.sum(read_labels[rmask] * controls[...,0].float()[rmask])
        write_loss =  torch.sum(write_labels[wmask] * controls[...,1].float()[wmask])
        controlling_loss = - (read_loss + write_loss) / controls.size(-1)
        return controlling_loss

    def compute_regul_loss(self, controls):
        Tt, B, Ts, _ = controls.size()  # Tt-1
        write_logits = controls[...,1]
        # read_proba = torch.exp(controls[...,0])
        # print('Writing decisions:', torch.exp(write_logits)[:, 0].gt(0.5).int().data)
        print('Writing decisions:', torch.exp(write_logits)[:, 0].data)

        # Minimize writing decisions below the diagonal
        # mask_below_diag = torch.tril(write_proba.new_ones(Tt, Ts), diagonal=-2)
        # # print('Mask below diag:', mask_below_diag)
        # mask_write = write_proba * mask_below_diag.unsqueeze(1)
        # mask_read  = read_proba * mask_below_diag.unsqueeze(1)
        # regul_loss_1 = - mask_write.sum() + mask_read.sum()
        # # print('Regul loss 1:', regul_loss_1)

        # Maximize writing deicision close to the diagonal
        mask_close_diag = torch.triu(write_logits.new_ones(Tt, Ts), diagonal=-1)
        mask_close_diag = torch.tril(mask_close_diag, diagonal=3)
        # print('Mask close diag:', mask_close_diag)
        mask_write = write_logits * mask_close_diag.unsqueeze(1)
        # mask_read  = read_proba * mask_close_diag.unsqueeze(1)
        regul_loss_2 = - mask_write.sum() 
        print('Regul loss 2:', regul_loss_2)
        # return regul_loss_2 + regul_loss_1
        return regul_loss_2



    def forward(self, sample, emissions, controls, gamma, read_labels, write_labels):
        """
        Emissions: B, Tt, Ts
        Controls: Tt-1, B, Ts, 2
        Gamma: Tt, B, Ts
        Read/Write: Tt-1, B, Ts
        """
        writing_loss = self.compute_writing_loss(gamma, emissions)
        controlling_loss = self.compute_control_loss(controls, read_labels, write_labels)
        regul_loss = self.compute_regul_loss(controls)
        # Total
        loss = writing_loss + self.alpha * controlling_loss + self.beta * regul_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data.item(),
            'writing_loss': writing_loss.data.item(),
            'nll_loss': writing_loss.data.item(),
            'controlling_loss':  controlling_loss.data.item(),
            'regul_loss':  regul_loss.data.item(),
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
        controlling_loss_sum = sum(log.get('controlling_loss', 0) for log in logging_outputs)
        regul_loss_sum = sum(log.get('regul_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'writing_loss': writing_loss_sum / sample_size / math.log(2),
            'nll_loss': writing_loss_sum / sample_size / math.log(2),
            'controlling_loss': controlling_loss_sum / sample_size / math.log(2),
            'regul_loss': regul_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
