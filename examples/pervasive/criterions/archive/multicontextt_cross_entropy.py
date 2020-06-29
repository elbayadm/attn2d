# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('multicontext_cross_entropy')
class MulticontextCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        # fmt: on

    def forward(self, model, sample, step=-1, epoche=-1, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss_multi(model, net_output, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss_multi(self, model, net_output, sample, reduce=True):
        step_logits = net_output[0]
        total_nll_loss = step_logits[0].new_zeros(1).float()
        total_loss = step_logits[0].new_zeros(1).float()

        Tt = len(step_logits)
        target = model.get_targets(sample, net_output)
        for t, logits in enumerate(step_logits):
            ttgt = target[:, t:t+1]
            lprobs =  utils.log_softmax(logits, dim=-1)
            B, ctx, V = lprobs.size()
            lprobs = lprobs.view(-1, lprobs.size(-1))
            ttgt = ttgt.unsqueeze(-1).expand(-1, -1, ctx)
            ttgt = ttgt.contiguous().view(-1, 1)
            non_pad_mask = ttgt.ne(self.padding_idx)
            nll_loss = -lprobs.gather(dim=-1, index=ttgt)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
            total_loss += loss/ctx
            total_nll_loss += nll_loss/ctx
        return total_loss, total_nll_loss


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
