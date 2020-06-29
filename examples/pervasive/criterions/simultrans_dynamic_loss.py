import logging
import math
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('simultrans_dynamic_loss')
class SimulTransDynamicLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.eps = args.label_smoothing
        self.alpha = args.write_scale
        self.beta = args.control_scale
        self.sentence_avg = args.sentence_avg

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--control-scale', default=1., type=float)
        parser.add_argument('--write-scale', default=1, type=float)

    def compute_writing_loss(self, Gamma, lprobs, target):
        """
        Target in B, Tt
        Lprobs logp (y_t | ..) in B, Tt, Ts
        Gamma select positions in B, Tt, Ts
        """
        B, Tt, Ts, V = lprobs.size()
        lprobs = utils.log_softmax(lprobs, dim=-1)
        target = target.unsqueeze(-1)  # B, Tt, 1
        non_pad_mask = target.ne(self.padding_idx)  # B, Tt, 1
        smooth_loss = -lprobs.sum(dim=-1)  # B, Tt, Ts
        # Gather ground truth:
        lprobs = lprobs.contiguous().view(-1, V)
        target = target.repeat(1, 1, Ts) # B,Tt,Ts
        nll_loss = -lprobs.gather(dim=-1, index=target.view(-1,1)).view(B, Tt, Ts)
        # Normalize by #contexts
        mask = Gamma.ne(0) * non_pad_mask
        denom = mask.sum(-1, keepdim=True).type_as(lprobs)  # B,Tt,1
        denom += (~non_pad_mask).type_as(denom)
        nll_loss = (Gamma * nll_loss / denom)[mask].sum()
        smooth_loss = (Gamma * smooth_loss/ denom)[mask].sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_control_loss(self, RWlogits, read_labels, write_labels):
        # controller
        Ts = RWlogits.size(2)
        rmask = torch.isfinite(RWlogits[..., 0])
        wmask = torch.isfinite(RWlogits[..., 1])
        read_loss =  torch.sum(read_labels[rmask] * RWlogits[...,0].float()[rmask]) 
        write_loss =  torch.sum(write_labels[wmask] * RWlogits[...,1].float()[wmask])
        controlling_loss = - (read_loss + write_loss) / RWlogits.size(-1)
        accuracy = (write_labels[wmask].eq(1) == RWlogits[...,1].float()[wmask].exp().gt(0.5)).float()
        positions = accuracy.numel()
        accuracy = accuracy.sum().long()
        return controlling_loss, accuracy, positions

    def forward(self, sample, lprobs, controller_out):
        """
        Lprobs : Tt, B, Tt, V
        Controls: Tt-1, B, Ts, 2
        Gamma: Tt, B, Ts
        Read/Write: Tt-1, B, Ts
        """
        Gamma, RWlogits, read_labels, write_labels = controller_out
        writing_loss, nll_loss = self.compute_writing_loss(Gamma, lprobs, sample['target'])
        controlling_loss, accuracy, positions = self.compute_control_loss(RWlogits, read_labels, write_labels)
        # Total
        loss = self.alpha * writing_loss + self.beta * controlling_loss 
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data.item(),
            'writing_loss': writing_loss.data.item(),
            'nll_loss': nll_loss.data.item(),
            'controlling_loss':  controlling_loss.data.item(),
            'accuracy':  accuracy.data.item(),
            'positions': positions,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """ Aggregate logging outputs from data parallel training. """
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        writing_loss_sum = sum(log.get('writing_loss', 0) for log in logging_outputs)
        controlling_loss_sum = sum(log.get('controlling_loss', 0) for log in logging_outputs)
        accuracy_sum = sum(log.get('accuracy', 0) for log in logging_outputs)

        npositions = sum(log.get('positions', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('writing_loss', writing_loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('controlling_loss', controlling_loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('accuracy', accuracy_sum / npositions, npositions, round=3)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
