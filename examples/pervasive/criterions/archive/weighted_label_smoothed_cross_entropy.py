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


@register_criterion('weighted_label_smoothed_cross_entropy')
class WeightedLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.weight_scale = args.weight_scale

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--weight-scale', default='inverse', type=str)

        # fmt: on

    def forward(self, model, sample, step=-1, epoche=-1, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if self.weight_scale in ['inverse', 'sqrt', 'cubic-root']:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        elif self.weight_scale == 'propto':
            loss, nll_loss = self.compute_propto_loss(model, net_output, sample, reduce=reduce)

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
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # B, T, V
        # scale:
        T = lprobs.size(1)
        if self.weight_scale == 'inverse':
            scales = 1 / torch.arange(1, T+1).float()
        elif self.weight_scale == 'sqrt':
            scales = torch.sqrt(torch.arange(1, T+1).float())
        elif self.weight_scale == 'cubic-root':
            scales = torch.arange(1, T+1).float() ** (1/3)

        # Sum to T as in the uniform
        scales = scales / scales.sum() * T
        scales = scales.unsqueeze(0).unsqueeze(-1).type_as(lprobs).to(lprobs.device)  # 1, T, V

        # print('scales:',scales.size(), scales)
        lprobs = lprobs * scales
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_propto_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # B, T, V
        B, T, V = lprobs.size()
        lprobs = lprobs.view(-1, lprobs.size(-1))  # BT, V
        target = model.get_targets(sample, net_output).view(-1, 1)  # BT
        # nll_loss_terms = -lprobs.gather(dim=-1, index=target).view(B, T)
        smooth_loss_terms = -lprobs.sum(dim=-1, keepdim=True).view(B, T)
        
        # nll_scales = nll_loss_terms.detach()  # B, T
        # # print('NLL scales:', nll_scales)
        # nll_scales = nll_scales[:, :1] / (nll_scales + 1e-5)
        # print('nll scales (l_1/l_t):', nll_scales)
        # nll_scales = nll_scales.view(-1, 1)

        smooth_ll_scales = smooth_loss_terms.detach()  # B, T
        smooth_ll_scales = smooth_ll_scales[:, :1] / (smooth_ll_scales + 1e-5)
        print('Smooth LL scales (l_1/l_t):', smooth_ll_scales)
        smooth_ll_scales = smooth_ll_scales.view(-1, 1)

        # Scales then ignore padding positions:
        non_pad_mask = target.ne(self.padding_idx)
        # nll_loss = -(lprobs * nll_scales).gather(dim=-1, index=target)[non_pad_mask]
        nll_loss = -(lprobs * smooth_ll_scales).gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -(lprobs * smooth_ll_scales).sum(dim=-1, keepdim=True)[non_pad_mask]
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
