"""
Importance sampling of the sentence smoothed loss:
    Loss = E_r[-log p] = E_q[-r/q log p]
    r referred to as scorer
    q referred to as sampler
    prior to normalization : q = ~q / Z_q and r = ~r / Z_r
    except fot q=hamming, the Z_q is untractable
    the importance ratios  w = r/q are approximated
    ~ w = ~r / ~q / (sum ~r / ~ q over MC)
    or
    ~ w = ~r / q / (sum ~r / q over MC)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .word import WordSmoothCriterion
from .samplers import init_sampler
from .scorers import init_scorer
from .utils import decode_sequence, get_ml_loss


class ImportanceSampler(nn.Module):
    """
    Apply sentence level loss smoothing
    with importance sampling
    q=p_\theta or hamming
    r = Cider or Bleu
    """
    def __init__(self, opt, loader):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.penalize_confidence = opt.penalize_confidence
        self.lazy_rnn = opt.lazy_rnn
        self.alpha = opt.alpha_sent
        self.mc_samples = opt.mc_samples
        self.combine_loss = opt.combine_loss
        self.sampler = init_sampler(opt.importance_sampler.lower(),
                                    opt,
                                    loader)
        self.vocab = loader.get_vocab()
        self.scorer = init_scorer(opt.reward.lower(),
                                  opt, self.vocab)
        if self.combine_loss:
            self.loss_sampled = WordSmoothCriterion(opt)
            # self.loss_gt = WordSmoothCriterion(opt)
            self.loss_gt = self.loss_sampled


    def log(self):
        self.logger.info('using importance sampling r=%s and q=%s' % (self.scorer.version,
                                                                      self.sampler.version))
        if self.combine_loss:
            self.logger.info('GT loss:')
            self.loss_gt.log()
            self.logger.info('Sampled loss:')
            self.loss_sampled.log()

    def forward(self, model,
                src_emb, src_code, state,
                input_lines_trg, trg_lengths,
                output_lines_trg,
                mask, scores=None):
        ilabels = input_lines_trg
        olabels = output_lines_trg
        logp = model.forward_decoder(src_emb, src_code, state,
                                     ilabels,
                                     trg_lengths)
        # Remove BOS token
        seq_length = logp.size(1)
        target = olabels[:, :seq_length]
        del olabels, ilabels
        mask = mask[:, :seq_length]
        if scores is not None:
            print('scaling the masks')
            # FIXME see to it that i do not normalize with scaled masks
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)

        # GT loss
        # okay only if loss_gt == loss_sampler (default) FIXME
        loss_gt, stats = self.batch_loss_lazy(logp, target, mask, scores)

        # Sampling loss
        monte_carlo = []
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for mci in range(MC):
            ipreds_matrix, opreds_matrix, sampled_q, stats = self.sampler.nmt_sample(logp, target) #FIXME
            # sampled, sampled_q, stats = self.sampler.sample(logp, target)
            # Score the sampled captions
            sampled_rewards, rstats = self.scorer.get_scores(opreds_matrix, target)
            stats.update(rstats)
            importance = sampled_rewards / sampled_q
            monte_carlo.append([ipreds_matrix, opreds_matrix, importance])
            # normalize:
        if MC == 1:
            # incorrect estimation of Z_r/Z_q
            monte_carlo = monte_carlo[0]
            importance_normalized = monte_carlo[2] / np.mean(monte_carlo[2])
            stats['importance_mean'] = np.mean(importance_normalized)
            stats['importance_std'] = np.std(importance_normalized)
            importance_normalized = Variable(torch.from_numpy(importance_normalized).float(),
                                             requires_grad=False).cuda().view(-1, 1)
            if self.lazy_rnn:
                mc_output, stats_sampled = self.batch_loss_lazy(logp,
                                                                monte_carlo[1],
                                                                mask,
                                                                importance_normalized)
            else:
                # Forward the sampled captions
                mc_output, stats_sampled = self.batch_loss(model,
                                                           src_emb,
                                                           src_code, state,
                                                           monte_carlo[0],
                                                           trg_lengths,
                                                           monte_carlo[1],
                                                           mask,
                                                           importance_normalized)
            if stats_sampled is not None:
                stats.update(stats_sampled)
            output = mc_output
        else:
            # correct estimation of Z_r/Z_q
            imp = np.vstack([_[2] for _ in monte_carlo]).T
            imp = imp/imp.sum(axis=1)[:, None]
            stats['importance_mean'] = np.mean(imp)
            stats['importance_std'] = np.std(imp)
            for mci in range(MC):
                importance_normalized = imp[:, mci]
                importance_normalized = Variable(torch.from_numpy(importance_normalized).float(),
                                                 requires_grad=False).cuda().view(-1,1)
                # print('imp:', list(imp[:, mci].T))
                if self.lazy_rnn:
                    mc_output, stats_sampled = self.batch_loss_lazy(logp,
                                                                    monte_carlo[mci][1],
                                                                    mask,
                                                                    importance_normalized)
                else:
                    mc_output, stats_sampled = self.batch_loss(model,
                                                               src_emb,
                                                               src_code, state,
                                                               monte_carlo[mci][0],
                                                               trg_lengths,
                                                               monte_carlo[mci][1],
                                                               mask,
                                                               importance_normalized)

                if stats_sampled is not None:
                    stats.update(stats_sampled)
                if not mci:
                    output = mc_output
                else:
                    output += mc_output
            output /= MC
        return loss_gt, output, stats

    def batch_loss(self, model, src_emb, src_code, state,
                   ipreds_matrix, trg_lengths, opreds_matrix,
                   mask, scores):
        """
        forward the new sampled labels and return the loss
        """
        logp = model.forward_decoder(src_emb, src_code, state,
                                     ipreds_matrix,
                                     trg_lengths)

        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              opreds_matrix,
                                              mask,
                                              scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)
        else:
            ml = get_ml_loss(logp,
                             opreds_matrix,
                             mask,
                             scores,
                             penalize=self.penalize_confidence)
            loss = ml
            stats = None
        return loss, stats

    def batch_loss_lazy(self, logp, target, mask, scores):
        """
        Evaluate the loss of the new labels given the gt logits
        """
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp, target, mask, scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)

        else:
            ml = get_ml_loss(logp, target, mask, scores)
            loss = ml
            stats = None
        return loss, stats





