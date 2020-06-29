import math
import torch
from fairseq import metrics, utils

from fairseq.criterions import FairseqCriterion, register_criterion

# Grid cross-entropy of pervasive attention models for simultaneous translation
# pa_grid_cross_entropy_v2


@register_criterion('grid_cross_entropy')
class GridCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.eps = args.label_smoothing
        self.lower_diag = args.lower_diag
        self.finish_reading = args.finish_reading
        self.dynamic_denom = args.dynamic_denom
        self.sentence_avg = args.sentence_avg

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        parser.add_argument('--lower-diag', default=0, type=int)
        parser.add_argument('--finish-reading', default=False, action='store_true')
        parser.add_argument('--dynamic-denom', default=False, action='store_true')

    def forward(self, model, sample, **kwargs):
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'nll_loss': utils.item(nll_loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        B, Tt, Ts, V = lprobs.size()
        target = model.get_targets(sample, net_output).unsqueeze(-1).repeat(1, 1, Ts) # B,Tt,Ts
        non_pad_mask = target.ne(self.padding_idx)
        grid_mask = torch.triu(target.new_ones(Tt, Ts), self.lower_diag)
        # Always include Ts in the contexts
        grid_mask[:,-1] = 1

        if self.finish_reading:
            shift_back = target.eq(self.padding_idx).int().sum(dim=-1).max().item() + 1
            # Last rows only with full context to predict EOS:
            grid_mask[-shift_back:,:-1] = 0
            grid_mask[-shift_back:,-1] = 1
        
        if self.dynamic_denom:
            denom = grid_mask.sum(-1, keepdim=True).unsqueeze(-1) # B,Tt,1,1
            if denom.eq(0).any():
                print('Denom:', denom.flatten())
                print('Tt,Ts=', Tt, Ts)
                print('Grid mask:', grid_mask[0])
                print('npad mask:', non_pad_mask)
                
            # Normalize by number of contexts
            lprobs = lprobs / denom.type_as(lprobs)
        else:
            # denom = (Ts - torch.arange(Tt)).clamp(min=1)
            denom = grid_mask.sum(dim=-1)
            lprobs = lprobs / denom.view(1,Tt,1,1).type_as(lprobs)

        grid_mask = grid_mask.repeat(B, 1, 1)
        select_mask = non_pad_mask * grid_mask.type_as(non_pad_mask)
        lprobs = lprobs.view(-1, V)
        select_mask = select_mask.view(-1, 1)
        target = target.view(-1, 1)
        nll_loss = -lprobs.gather(dim=-1, index=target)[select_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[select_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """ Aggregate logging outputs from data parallel training. """
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
