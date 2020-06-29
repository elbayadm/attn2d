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


@register_criterion('simultaneous_translation_loss')
class simultaneous_translation_loss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.control_scale
        self.eps = 1e-6
        self.delay_margin = args.delay_margin
        self.without_mg = args.without_marginalizing

    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--control-scale', default=1., type=float, metavar='D')
        parser.add_argument('--delay-scale', default=0., type=float, metavar='D')
        parser.add_argument('--delay-margin', default=0., type=float, metavar='D')
        parser.add_argument('--without-marginalizing', action='store_true')

    def forward(self, sample, lprobs, gamma, controls, read, write):
        """
        Emissions: Tt, N, Ts
        Controls: Tt-1, N, Ts
        Gamma: Tt, N, Ts
        Read/Write: Tt-1, N, Ts
        """
        Tt, N, Ts = lprobs.size()
        # writer
        if self.without_mg:
            mask = torch.isfinite(lprobs)
            writing_loss = - torch.sum(lprobs.float()[mask]) / (N*Tt)
        else:
            gamma = gamma.float()
            mask = gamma.ne(0)
            writing_loss = - torch.sum(gamma[mask] * lprobs.float()[mask])

        # controller
        read = read.float()
        write = write.float()
        print('Read labels:', read[:, 0])
        print('Write labels:', write[:, 0])
        # print('Read controls:', torch.exp(controls[:, 0, :, 0].float()))
        # print('Write controls:', torch.exp(controls[:, 0, :, 1].float()))

        # rmask = read.ne(0)
        # wmask = write.ne(0)
        rmask = torch.isfinite(controls[..., 0])
        wmask = torch.isfinite(controls[..., 1])
        read_loss =  torch.sum(read[rmask] * controls[...,0].float()[rmask])
        write_loss =  torch.sum(write[wmask] * controls[...,1].float()[wmask])
        controlling_loss = - read_loss - write_loss
        if self.delay_margin:
            # Regularize the reading:
            regul_term = self.delay_margin * torch.exp(controls[...,0].float()[rmask]).sum()
            controlling_loss += regul_term
            # print('Regul_term:', regul_term)
        # print('W:', write_loss, 'R:', read_loss)
        # Total
        loss = writing_loss + self.alpha * controlling_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # print('Loss:', loss, 'Writing:', writing_loss, 'Control loss:', controlling_loss)
        logging_output = {
            'loss': loss.data.item(),
            'writing_loss': writing_loss.data.item(),
            'nll_loss': writing_loss.data.item(),

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
        controlling_loss_sum = sum(log.get('controlling_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'writing_loss': writing_loss_sum / sample_size / math.log(2),
            'nll_loss': writing_loss_sum / sample_size / math.log(2),
            'controlling_loss': controlling_loss_sum / sample_size / math.log(2),
            'gamma': sum([log.get('gamma', 0) for log in logging_outputs]) / nsentences,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
