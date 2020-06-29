# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion

# Grid cross-entropy of pervasive attention models

@register_criterion('pa_grid_cross_entropy')
class PAGridCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lower_diag = args.lower_diag
        self.upper_diag = args.upper_diag

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
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        B, Tt, Ts, V = lprobs.size()
        target = model.get_targets(sample, net_output).unsqueeze(-1).repeat(1, 1, Ts) # B,Tt,Ts
        non_pad_mask = target.ne(self.padding_idx)
        # A select area of the grid
        grid_mask = torch.triu(target.new_ones(Tt, Ts), -self.lower_diag-(Tt-Ts)*(Tt>Ts))
        grid_mask = torch.tril(grid_mask, self.upper_diag+(Ts-Tt)*(Ts>Tt))
        grid_mask = grid_mask.repeat(B, 1, 1)
        select_mask = non_pad_mask * grid_mask.type_as(non_pad_mask)
        denom = grid_mask.sum(-1, keepdim=True).unsqueeze(-1) # B,Tt,1,1
        # if denom.eq(0).any():
            # print('Denom:', denom.flatten())
            # print('Tt,Ts=', Tt, Ts)
            # print('Grid mask:', grid_mask)
            # print('npad mask:', non_pad_mask)
        # Normalize by number of contexts
        lprobs = lprobs / denom.type_as(lprobs)
        lprobs = lprobs.view(-1, V)
        select_mask = select_mask.view(-1, 1)
        target = target.view(-1, 1)
        nll_loss = -lprobs.gather(dim=-1, index=target)[select_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[select_mask]
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
