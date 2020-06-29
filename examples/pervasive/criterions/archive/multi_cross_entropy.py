# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('multi_cross_entropy')
class MultiCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.scale = args.loss_scale

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--loss-scale', default='uniform', type=str)

        # fmt: on

    def get_loss_scale(self, n):
        if self.scale == 'uniform':
            return 1
        elif self.scale == 'inverse':
            return 1/n
        elif self.scale == 'inverse_sqrt':
            return 1/math.sqrt(n)
        elif self.scale == 'prop':
            return n
        else:
            raise ValueError('Unknonw scaling ', self.scale)

    def forward(self, model, sample, step=-1, epoche=-1, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        decoder_outputs = model(**sample['net_input'])
        total_nll_loss = decoder_outputs[0].new_zeros(1).float()
        total_loss = decoder_outputs[0].new_zeros(1).float()
        total_scales = 0
        for i, output in enumerate(decoder_outputs):
            loss, nll_loss = self.compute_loss(output, sample, reduce=reduce)
            scale = self.get_loss_scale(i+1)
            total_scales += scale
            total_loss = total_loss + scale * loss
            total_nll_loss = total_nll_loss + scale * nll_loss
        # Average:
        total_loss = total_loss / total_scales
        total_nll_loss = total_nll_loss / total_scales

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'nll_loss': utils.item(total_nll_loss.data) if reduce else total_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        lprobs = utils.log_softmax(net_output, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1, 1)
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
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
