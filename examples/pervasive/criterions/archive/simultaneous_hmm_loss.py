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


@register_criterion('simultaneous_hmm_loss')
class Simultaneous_hmm_loss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.control_scale
        self.eps = 1e-6
        self.delay_margin = args.delay_margin

    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--control-scale', default=1., type=float, metavar='D')
        parser.add_argument('--delay-scale', default=0., type=float, metavar='D')
        parser.add_argument('--delay-margin', default=0., type=float, metavar='D')

    def forward(self, sample, lprobs, controls, marginals, reduce=True):
        N, Tt, Ts = lprobs.size()
        # writer
        gamma = marginals['gamma'].float()
        writing_loss = - torch.sum(gamma * lprobs)
        if 'read/write' in marginals:  # Trainable controller
            read = marginals['read/write'][0].float()
            write = marginals['read/write'][0].float()
            controlling_loss = - torch.sum((read * controls[:, :-1, :, 0] + write * controls[:, :-1, :, 1]))
            loss = writing_loss + self.alpha * controlling_loss
        else:
            controlling_loss = writing_loss.new_zeros(1)
            loss = writing_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        gamma = gamma.sum(dim=0).sum(dim=0)  # Ts
        print('Gamma:', gamma)
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
