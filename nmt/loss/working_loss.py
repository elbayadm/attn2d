import os
import os.path as osp
import sys
import random
import collections
import math
import pickle
from scipy.special import binom
import numpy as np
from scipy.spatial.distance import hamming
from collections import Counter, OrderedDict
from scipy.misc import comb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from misc.utils import to_contiguous, decode_sequence, sentence_bleu, group_similarity

_BOS = 2
_EOS = 1


def hamming_distrib_soft(m, v, tau):
    x = [np.log(comb(m, d, exact=False)) + d * np.log(v) - d/tau * np.log(v) - d/tau for d in range(m + 1)]
    x = np.array(x)
    p = np.exp(x)
    p /= np.sum(p)
    return p


def hamming_distrib(m, v, tau):
    x = [comb(m, d, exact=False) * (v-1)**d * math.exp(-d/tau) for d in range(m+1)]
    x = np.array(x)
    Z = np.sum(x)
    return x/Z, Z


def hamming_Z(m, v, tau):
    pd = hamming_distrib(m, v, tau)
    popul = v ** m
    print('popul:', popul)
    print('pd:', pd)
    Z = np.sum(pd * popul * np.exp(-np.arange(m+1)/tau))
    print('Pre-clipping:', Z)
    return np.clip(Z, a_max=1e30, a_min=1)


def rows_entropy(distrib):
    """
    return the entropy of each row in the given distributions
    """
    return torch.sum(distrib * torch.log(distrib), dim=1)


def normalize_reward(distrib):
    """
    Normalize so that each row sum to one
    """
    sums = torch.sum(distrib, dim=1).unsqueeze(1)
    return  distrib / sums.repeat(1, distrib.size(1))


class LanguageModelCriterion(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.scale_loss = opt.scale_loss
        self.logger.warn('Initiating ML loss')

    def forward(self, input, target, mask, scores=None):
        """
        input : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.unsqueeze(1).repeat(1, seq_length)
            # print('mask:', mask.size(), 'row_scores:', row_scores.size())
            mask = torch.mul(mask, row_scores)
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output, output, None


def get_ml_loss(input, target, mask, norm=None):
    """
    Compute the usual ML loss
    """
    input = to_contiguous(input).view(-1, input.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    ml_output = - input.gather(1, target) * mask
    if torch.sum(mask).data[0] > 0:
        ml_output = torch.sum(ml_output) / torch.sum(mask)
    else:
        raise ValueError('Mask shouldnt be all null')
    return ml_output

def get_indices_vocab(target, seq_per_img):
    seq_length = target.size(1)
    num_img = target.size(0) // seq_per_img
    vocab_per_image = target.chunk(num_img)
    vocab_per_image = [np.unique(to_contiguous(t).data.cpu().numpy())
                       for t in vocab_per_image]
    max_vocab = max([len(t) for t in vocab_per_image])
    vocab_per_image = [np.pad(t, (0, max_vocab - len(t)), 'constant')
                       for t in vocab_per_image]
    indices_vocab = Variable(torch.cat([torch.from_numpy(t).\
                             repeat(seq_per_img * seq_length, 1)
                             for t in vocab_per_image], dim=0)).cuda()
    return indices_vocab


class WordSmoothCriterion2(nn.Module):
    """
    Apply word level loss smoothing given a similarity matrix
    the two versions are:
        full : to take into account the whole vocab
        limited: to consider only the ground truth vocab
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.scale_loss = opt.scale_loss
        self.smooth_remove_equal = opt.smooth_remove_equal
        self.clip_sim = opt.clip_sim
        self.add_entropy = opt.word_add_entropy
        self.normalize_batch = opt.normalize_batch
        self.scale_wl = opt.scale_wl
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_word
        # assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        M = M - 1
        if opt.rare_tfidf:
            IDF = pickle.load(open('data/coco/idf_coco.pkl', 'rb'))
            M += self.tau_word * IDF  # old versions IDF/1.8
        M = M.astype(np.float32)
        n, d = M.shape
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        self.vocab_size = opt.vocab_size
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M
        self.exact = opt.exact_dkl

    def log(self):
        self.logger.info("Initialized Word2 loss tau=%.3f, alpha=%.1f" % (self.tau_word, self.alpha))

    def forward(self, input, target, mask, scores=None):
        # truncate to the same size
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask,
                                norm=self.normalize_batch)
        # Get the similarities of the words in the batch (NxL, V)
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        sim = self.Sim_Matrix[indices]
        # print('raw sim:', sim)
        if self.clip_sim:
            # keep only the similarities larger than the margin
            # self.logger.warn('Clipping the sim')
            sim = sim * sim.ge(self.margin).float()
        if self.limited:
            # self.logger.warn('Limitig smoothing to the gt vocab')
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            input = input.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(sim, 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = sim
        # Normalize the word reward distribution:
        smooth_target = normalize_reward(smooth_target)

        if self.exact:
            delta = Variable(torch.eye(self.vocab_size)[indices.cpu()]).cuda()
            smooth_target = torch.mul(smooth_target, self.alpha) + torch.mul(delta, (1 - self.alpha))
            # print("Smooth:", smooth_target)

        # Store some stats about the sentences scores:
        scalars = smooth_target.data.cpu().numpy()
        # print("Reward multip:", scalars[0][:10])

        stats = {"word_mean": np.mean(scalars),
                 "word_std": np.std(scalars)}

        # print('smooth_target:', smooth_target)
        # Format
        mask = to_contiguous(mask).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        # print('in:', input.size(), 'mask:', mask.size(), 'smooth:', smooth_target.size())
        output = - input * mask.repeat(1, sim.size(1)) * smooth_target

        if self.normalize_batch:
            if torch.sum(mask).data[0] > 0:
                output = torch.sum(output) / torch.sum(mask)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
        else:
            output = torch.sum(output)

        if self.add_entropy:
            H = rows_entropy(smooth_target).unsqueeze(1)
            entropy = torch.sum(H * mask)
            if self.normalize_batch:
                entropy /= torch.sum(mask)
            # print('Entropy:', entropy.data[0])
            output += entropy

        if self.scale_wl:
            self.logger.warn('Scaling the pure WL RAML by %.3f' % self.scale_wl)
            output = self.scale_wl * output
        output = self.alpha * output + (1 - self.alpha) * ml_output
        return ml_output, output, stats


class WordSmoothCriterion(nn.Module):
    """
    Apply word level loss smoothing given a similarity matrix
    the two versions are:
        full : to take into account the whole vocab
        limited: to consider only the ground truth vocab
    """
    def __init__(self, opt, vocab):
        super().__init__()
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.scale_loss = opt.scale_loss
        self.smooth_remove_equal = opt.smooth_remove_equal
        self.clip_sim = opt.clip_sim
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        M = M.astype(np.float32)
        n, d = M.shape
        assert n == d and n == len(vocab), 'Similarity matrix has incompatible shape'
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M

    def forward(self, input, target, mask, scores=None):
        # truncate to the same size
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        # Get the similarities of the words in the batch (Vb, V)
        sim = self.Sim_Matrix[to_contiguous(target).view(-1, 1).squeeze().data]
        # print('raw sim:', sim)
        if self.clip_sim:
            # keep only the similarities larger than the margin
            # self.logger.warn('Clipping the sim')
            sim = sim * sim.ge(self.margin).float()
        if self.limited:
            # self.logger.warn('Limitig smoothing to the gt vocab')
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            input = input.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(torch.add(sim, -1.), 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = torch.add(sim, -1.)
        if self.smooth_remove_equal:
            smooth_target = smooth_target * sim.lt(1.0).float()
        # print('smooth_target:', smooth_target)
        # Store some stats about the sentences scores:
        scalars = smooth_target.data.cpu().numpy()[:]
        stats = {"word_mean": np.mean(scalars),
                 "word_std": np.std(scalars)}
        # Format
        mask = to_contiguous(mask).view(-1, 1)
        mask = mask.repeat(1, sim.size(1))
        input = to_contiguous(input).view(-1, input.size(2))
        # print('in:', input.size(), 'mask:', mask.size(), 'smooth:', smooth_target.size())
        output = - input * mask * smooth_target

        if torch.sum(smooth_target * mask).data[0] > 0:
            output = torch.sum(output) / torch.sum(smooth_target * mask)
        else:
            self.logger.warn("Smooth targets weights sum to 0")
            output = torch.sum(output)

        return ml_output, self.alpha * output + (1 - self.alpha) * ml_output, stats


class SentSmoothCriterion(nn.Module):
    """
    Apply sentence level loss smoothing
    """
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        # TODO assert type is defined
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        # tau set to zero means no exponentiation
        self.tau_sent = opt.tau_sent
        self.scale_loss = opt.scale_loss

    def forward(self, input, target, mask, scores=None):
        # truncate
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)
        # Process scores:
        if self.tau_sent:
            sent_scores = np.exp(np.array(sent_scores) / self.tau_sent)
        else:
            sent_scores = np.array(sent_scores)
            if not np.sum(sent_scores):
                self.logger.warn('Adding +1 to the zero scores')
                sent_scores += 1
        # sent_scores from (N, 1) to (N, seq_length)
        self.logger.warn('Scores after processing: %s' % str(sent_scores))
        # Store some stats about the sentences scores:
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}
        sent_scores = np.repeat(sent_scores, seq_length)
        smooth_target = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()
        # substitute target with the prediction (aka sampling wrt p_\theta)
        preds = Variable(preds[:, :seq_length]).cuda()
        # Flatten
        preds = to_contiguous(preds).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, preds) * mask * smooth_target
        if torch.sum(smooth_target * mask).data[0] > 0:
            output = torch.sum(output) / torch.sum(smooth_target * mask)
        else:
            self.logger.warn("Smooth targets weights sum to 0")
            self.logger.warn('Mask: %s' % str(torch.sum(mask).data[0]))
            self.logger.warn('Scores: %s' % str(torch.sum(smooth_target).data[0]))
            # output = torch.sum(output)
            output = torch.zeros(1)

        return ml_output, self.alpha * output + (1 - self.alpha) * ml_output, stats


class WordSentSmoothCriterion(nn.Module):
    """
    Combine a word level smothing with sentence scores
    """
    def __init__(self, opt, vocab):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        self.scale_loss = opt.scale_loss
        self.clip_sim = opt.clip_sim
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha
        assert self.alpha > -2, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_sent = opt.tau_sent
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        M = M.astype(np.float32)
        n, d = M.shape
        assert n == d and n == len(vocab), \
                'Similarity matrix has incompatible shape %d x %d \
                whilst vocab size is %d' % (n, d, len(vocab))
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M

    def forward(self, input, target, mask, scores=None):
        # truncate
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        # Sentence level
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)
        # Process scores:
        if self.tau_sent:
            sent_scores = np.exp(np.array(sent_scores) / self.tau_sent)
        else:
            sent_scores = np.array(sent_scores)
            if not np.sum(sent_scores):
                self.logger.warn('Adding +1 to the zero scores')
                sent_scores += 1

        # sent_scores from (N, 1) to (N, seq_length)
        self.logger.warn('Scores after processing: %s' % str(sent_scores))
        # Store some stats about the sentences scores:
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}

        sent_scores = np.repeat(sent_scores, seq_length)
        smooth_target = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()

        # Word level
        preds = Variable(preds[:, :input.size(1)]).cuda()
        preds = to_contiguous(preds).view(-1, 1)
        sim = self.Sim_Matrix[preds.squeeze().data]
        if self.tau_word:
            smooth_target_wl = torch.exp(torch.mul(torch.add(sim, -1.), 1/self.tau_word))
        else:
            smooth_target_wl = torch.add(sim, -1.)

        scalars = smooth_target_wl.data.cpu().numpy()[:]
        stats["word_mean"] = np.mean(scalars)
        stats["word_std"] = np.std(scalars)

        mask_wl = mask.repeat(1, sim.size(1))
        # format the sentence scores
        smooth_target = smooth_target.repeat(1, sim.size(1))
        output_wl = - input * smooth_target_wl * mask_wl * smooth_target
        norm = torch.sum(smooth_target_wl * mask_wl * smooth_target)
        if norm.data[0] > 0:
            output = torch.sum(output_wl) / norm
        else:
            self.logger.warn("Smooth targets weights sum to 0")
            output = torch.sum(output_wl)
        return ml_output, self.alpha * output + (1 - self.alpha) * ml_output, stats


class CiderRewardCriterion(SentSmoothCriterion, WordSentSmoothCriterion):
    def __init__(self, opt, vocab):
        if 'word' in opt.loss_version:
            WordSentSmoothCriterion.__init__(self, opt, vocab)
        else:
            SentSmoothCriterion.__init__(self, opt)
        self.vocab = vocab

    def get_scores(self, preds, target):
        # The reward loss:
        cider_scorer = CiderScorer(n=4, sigma=6)
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        for e, h in enumerate(hypo):
            ix_start =  e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            cider_scorer += (h, refs[ix_start : ix_end])
        (score, scores) = cider_scorer.compute_score()
        self.logger.debug("CIDEr score: %s" %  str(scores))
        return scores

    def forward(self, input, target, mask, scores=None):
        if 'word' in self.version:
            return WordSentSmoothCriterion.forward(self, input, target, mask, scores)
        else:
            return SentSmoothCriterion.forward(self, input, target, mask, scores)


class BleuRewardCriterion(SentSmoothCriterion, WordSentSmoothCriterion):
    def __init__(self, opt, vocab):
        if 'word' in opt.loss_version:
            WordSentSmoothCriterion.__init__(self, opt, vocab)
        else:
            SentSmoothCriterion.__init__(self, opt)
        self.vocab = vocab
        self.bleu_order = int(self.version[-1])
        self.bleu_scorer = opt.bleu_version
        assert self.bleu_scorer in ['coco', 'soft'], "Unknown bleu scorer %s" % self.bleu_scorer

    def get_scores(self, preds, target):
        if self.bleu_scorer == 'coco':
            bleu_scorer = BleuScorer(n=self.bleu_order)
            coco = True
        else:
            coco = False
            scores = []
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        for e, h in enumerate(hypo):
            ix_start =  e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            if coco:
                bleu_scorer += (h, refs[ix_start : ix_end])
            else:
                scores.append(sentence_bleu(h, ' '.join(refs[ix_start: ix_end]),
                                            order=self.bleu_order))
        if coco:
            (score, scores) = bleu_scorer.compute_score()
            scores = scores[-1]
        self.logger.debug("Bleu scores: %s" %  str(scores))
        return scores

    def forward(self, input, target, mask, scores=None):
        if 'word' in self.version:
            return WordSentSmoothCriterion.forward(self, input, target, mask, scores)
        else:
            return SentSmoothCriterion.forward(self, input, target, mask, scores)


class InfersentRewardCriterion(SentSmoothCriterion, WordSentSmoothCriterion):
    def __init__(self, opt, vocab):
        if 'word' in opt.loss_version:
            WordSentSmoothCriterion.__init__(self, opt, vocab)
        else:
            SentSmoothCriterion.__init__(self, opt)
        self.vocab = vocab
        self.logger.info('loading the infersent pretrained model')
        glove_path = '../infersent/dataset/glove/glove.840b.300d.txt'
        self.infersent = torch.load('../infersent/infersent.allnli.pickle',
                                    map_location=lambda storage, loc: storage)
        self.infersent.set_glove_path(glove_path)
        self.infersent.build_vocab_k_words(k=100000)
        # freeze infersent params:
        for p in self.infersent.parameters():
            p.requires_grad = false

    def get_scores(self, preds, target):
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        scores = []
        lr = len(refs)
        codes = self.infersent.encode(refs + hypo)
        refs = codes[:lr]
        hypo = codes[lr:]
        for e, h in enumerate(hypo):
            ix_start =  e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            scores.append(group_similarity(h, refs[ix_start : ix_end]))
        self.logger.debug("infersent similairities: %s" %  str(scores))
        return scores

    def forward(self, input, target, mask, scores=None):
        if 'word' in self.version:
            return WordSentSmoothCriterion.forward(self, input, target, mask, scores)
        else:
            return SentSmoothCriterion.forward(self, input, target, mask, scores)


class HammingRewardCriterion(SentSmoothCriterion, WordSentSmoothCriterion):
    def __init__(self, opt, vocab):
        if 'word' in opt.loss_version:
            WordSentSmoothCriterion.__init__(self, opt, vocab)
        else:
            SentSmoothCriterion.__init__(self, opt)

    def get_scores(self, preds, target):
        refs = target.cpu().data.numpy()
        # Hamming distances
        scores = np.array([- hamming(u, v) for u, v in zip(preds.numpy(), refs)])
        self.logger.debug("Negative hamming distances: %s" %  str(scores))
        return scores

    def forward(self, input, target, mask, scores=None):
        if 'word' in self.version:
            return WordSentSmoothCriterion.forward(self, input, target, mask, scores)
        else:
            return SentSmoothCriterion.forward(self, input, target, mask, scores)


class RewardSampler(nn.Module):
    """
    Sampling the sentences wtr the reward distribution
    instead of the captionig model itself
    """
    def __init__(self, opt, vocab):
        super(RewardSampler, self).__init__()
        self.logger = opt.logger
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau = opt.tau_sent
        self.combine_loss = opt.combine_loss
        self.scale_loss = opt.scale_loss
        self.vocab_size = opt.vocab_size
        self.vocab = vocab
        self.limited = opt.limited_vocab_sub
        self.verbose = opt.verbose
        self.mc_samples = opt.mc_samples
        # print('Training:', self.training)
        if self.combine_loss:
            # Instead of ML(sampled) return WL(sampled)
            self.loss_sampled = WordSmoothCriterion2(opt)
            # self.loss_sampled.alpha = .7
            self.loss_gt = WordSmoothCriterion2(opt)

    def forward(self, model, input_lines_src, input_lines_trg,
                output_lines_trg, mask, scores=None):
        ilabels = input_lines_trg
        olabels = output_lines_trg
        # truncate
        input = model.forward(input_lines_src, ilabels)
        # Remove BOS token
        N = input.size(0)
        seq_length = input.size(1)
        target = olabels[:, :seq_length]
        del olabels, ilabels
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        # GT loss
        if self.combine_loss:
            ml_gt, loss_gt, stats_gt = self.loss_gt(input, target, mask)
        else:
            loss_gt = get_ml_loss(input, target, mask)
            ml_gt = loss_gt
        # Sampling loss
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for ss in range(MC):
            ipreds_matrix, opreds_matrix, stats = self.alter(target)
            # Forward the sampled captions
            sample_input = model.forward(input_lines_src, ipreds_matrix)
            if self.combine_loss:
                _, loss_sampled, stats_sampled = self.loss_sampled(sample_input, opreds_matrix, mask)

                stats.update(stats_sampled)
            else:
                loss_sampled = get_ml_loss(sample_input, opreds_matrix, mask)
            if not ss:
                output = loss_sampled
            else:
                output += loss_sampled
        output /= MC
        output = self.alpha * output + (1 - self.alpha) * loss_gt
        return ml_gt, output, stats


class HammingRewardSampler(RewardSampler):
    """
    Sample a hamming distance and alter the truth
    """
    def __init__(self, opt, vocab):
        RewardSampler.__init__(self, opt, vocab)

    def log(self):
        sl = "ML" if not self.combine_loss else "Word"
        self.logger.info('Initialized hamming reward sampler tau = %.2f, alpha= %.1f limited=%d sampled loss = %s' % (self.tau, self.alpha, self.limited, sl))

    def alter(self, labels):
        # lables tokens + EOS + padding
        # ipreds : BOS + tokens + padding
        # opreds : tokens + EOS + padding
        N = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        if self.limited:
            batch_vocab = np.delete(np.unique(refs), 0)
            lv = len(batch_vocab)
        else:
            lv = self.vocab_size
        self.logger.warn('sampling with V=%d:' % lv)
        distrib, Z = hamming_distrib(seq_length, lv, self.tau)
        if self.training:
            self.logger.debug('Sampling distrib (Z=%.3e) %s' % (Z, distrib))
        # Sample a distance i.e. a reward
        select = np.random.choice(a=np.arange(seq_length + 1),
                                  p=distrib)
        # score = math.exp(-select / self.tau)
        # score = distrib[select]
        score = math.exp(-select / self.tau) / Z
        if self.training:
            self.logger.debug("reward (d=%d): %.2e" %
                              (select, score))
        stats = {"sent_mean": score,
                 "sent_std": 0}

        # Format preds by changing d=select tokens at random
        preds = refs
        # choose tokens to replace
        change_index = np.random.randint(seq_length, size=(N, select))
        rows = np.arange(N).reshape(-1, 1).repeat(select, axis=1)
        # select substitutes
        if self.limited:
            select_index = np.random.choice(batch_vocab, size=(N, select))
        else:
            select_index = np.random.randint(low=4, high=self.vocab_size, size=(N, select))
        # print("Selected:", select_index)
        preds[rows, change_index] = select_index
        # print('opreds:', preds)
        # remove EOS to get ipreds:
        ipreds = preds[:, :-1]
        ipreds[ipreds == _EOS] = 0
        ipreds_matrix = np.hstack((np.ones((N, 1)) * _BOS, ipreds))  # padd <BOS>
        # print('ipreds:', ipreds_matrix)
        ipreds_matrix = Variable(torch.from_numpy(ipreds_matrix)).cuda().type_as(labels)
        opreds_matrix = Variable(torch.from_numpy(preds)).cuda().type_as(labels)
        return ipreds_matrix, opreds_matrix, stats


class MultiLanguageModelCriterion(nn.Module):
    def __init__(self, seq_per_img=5):
        super(MultiLanguageModelCriterion, self).__init__()
        self.seq_per_img = seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        max_length = input.size(1)
        num_img = input.size(0) // self.seq_per_img
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        real_output = torch.sum(output) / torch.sum(mask)
        # ------------------------------------------------
        output = output.view(-1, max_length)
        sent_scores = output.sum(dim=1) / mask_.sum(dim=1)
        sent_scores_per_image = sent_scores.chunk(num_img)
        output = torch.sum(torch.cat([t.max() for t in sent_scores_per_image], dim=0))
        output = output / num_img
        return real_output, output


class DataAugmentedCriterion(nn.Module):
    """
    Treat the augmented captions separately
    """
    def __init__(self, opt):
        super(DataAugmentedCriterion, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.seq_per_img = opt.seq_per_img
        assert self.seq_per_img > 5, 'Captions per image is seq than 5'
        # The GT loss
        if opt.gt_loss_version == 'word':
            self.crit_gt = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_gt = LanguageModelCriterion(opt)
            # Ensure loss scaling with the imprtance sampling ratios
            self.crit_gt.scale_loss = 0

        # The augmented loss
        if opt.augmented_loss_version == 'word':
            self.crit_augmented = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_augmented = LanguageModelCriterion(opt)
            # Ensure loss scaling with the imprtance sampling ratios
            self.crit_augmented.scale_loss = 1

    def forward(self, input, target, mask, scores):
        seq_length = input.size(1)
        batch_size = input.size(0)
        # truncate
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        # Separate gold from augmented
        num_img = batch_size // self.seq_per_img
        input_per_image = input.chunk(num_img)
        mask_per_image = mask.chunk(num_img)
        target_per_image = target.chunk(num_img)
        scores_per_image = scores.chunk(num_img)

        input_gt = torch.cat([t[:5] for t in input_per_image], dim=0)
        target_gt = torch.cat([t[:5] for t in target_per_image], dim=0)
        mask_gt = torch.cat([t[:5] for t in mask_per_image], dim=0)

        input_gen = torch.cat([t[5:] for t in input_per_image], dim=0)
        target_gen = torch.cat([t[5:] for t in target_per_image], dim=0)
        mask_gen = torch.cat([t[5:] for t in mask_per_image], dim=0)
        scores_gen = torch.cat([t[5:] for t in scores_per_image], dim=0)
        # print('Splitted data:', input_gt.size(), target_gt.size(), mask_gt.size(),
              # 'gen:', input_gen.size(), target_gen.size(), mask_gen.size(), scores_gen.size())

        # For the first 5 captions per image (gt) compute LM
        _, output_gt = self.crit_gt(input_gt, target_gt, mask_gt)

        # For the rest of the captions: importance sampling
        _, output_gen = self.crit_augmented(input_gen, target_gen, mask_gen, scores_gen)
        # TODO check if must combine with ml augmented as well
        return output_gt, self.beta * output_gen + (1 - self.beta) * output_gt


class PairsLanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(PairsLanguageModelCriterion, self).__init__()
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        #  print "target:", target
        #  print "mask:", mask
        # duplicate
        num_img = input.size(0) // self.seq_per_img
        input_per_image = input.chunk(num_img)
        input = torch.cat([t.repeat(self.seq_per_img, 1, 1) for t in input_per_image], dim=0)
        target = torch.unsqueeze(target, 0)
        target = target.permute(1, 0, 2)
        target = target.repeat(1, self.seq_per_img, 1)
        target = target.resize(target.size(0) * target.size(1), target.size(2))
        mask = mask[:, :input.size(1)]
        mask = torch.unsqueeze(mask, 0)
        mask = mask.permute(1, 0, 2)
        mask = mask.repeat(1, self.seq_per_img, 1)
        mask = mask.resize(mask.size(0) * mask.size(1), mask.size(2))
        #  print "target:", target
        #  print "mask:", mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output, output


