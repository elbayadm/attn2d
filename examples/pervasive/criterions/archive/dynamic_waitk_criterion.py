# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
# TODO : set back

import math
import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('dynamic_waitk')
class DynamicWaitkCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.alpha = args.context_scale

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--context-scale', default=1., type=float)
        # fmt: on

    def forward(self, logits, read, write, sample, reduce=True, ** kwargs):
        writing_loss, nll_loss = self.compute_writing_loss(logits, sample['target'])
        control_loss, accuracy, num_decisions = self.compute_control_loss(read, write)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # if self.alpha:
        loss = writing_loss + self.alpha * control_loss
        logging_output = {
            'writing_loss': utils.item(writing_loss.data),
            'controlling_loss': utils.item(control_loss.data),
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'num_decisions': utils.item(num_decisions),
            'accuracy': utils.item(accuracy),
        }
        return loss, sample_size, logging_output

    def compute_control_loss(self, read, write):
        control_loss = - (read.sum() + write.sum())
        accuracy = (read > math.log(0.5)).sum() + (write > math.log(0.5)).sum()
        return control_loss, accuracy, read.numel() + write.numel()

    def compute_writing_loss(self, lprobs, target, reduce=True):
        # print('Target:', target.size())
        # print('lprobs:', lprobs.size())
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_decisions = sum(log.get('num_decisions', 0) for log in logging_outputs)

        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'writing_loss': sum(log.get('writing_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'controlling_loss': sum(log.get('controlling_loss', 0) for log in logging_outputs) / num_decisions,
            'accuracy': sum(log.get('accuracy', 0) for log in logging_outputs) / float(num_decisions),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
