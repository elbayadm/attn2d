"""
Bleu reward scorer for importance sampling
"""
import math
from collections import Counter
import numpy as np
from ..utils import decode_sequence


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    """
    Compute sentence-level BLEU score between a translation hypothesis and a reference.

    :param hypothesis: list of tokens or token ids
    :param reference: list of tokens or token ids
    :param smoothing: apply smoothing (recommended, especially for short sequences)
    :param order: count n-grams up to this value of n.
    :param kwargs: additional (unused) parameters
    :return: BLEU score (float)
    """
    log_score = 0

    if len(hypothesis) == 0:
        return 0

    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))

        numerator = sum(min(count, ref_ngrams[bigram])
                        for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())

        if smoothing:
            numerator += 1
            denominator += 1

        score = numerator / denominator

        if score == 0:
            log_score += float('-inf')
        else:
            log_score += math.log(score) / order

    bp = min(1, math.exp(1 - len(reference) / len(hypothesis)))
    return math.exp(log_score) * bp


class BleuRewardScorer(object):
    """
    Evaluate Bleu scores of given sentences wrt gt
    """
    def __init__(self, opt, vocab):
        self.version = opt.reward
        self.vocab = vocab
        self.bleu_order = int(opt.reward[-1])
        self.seq_per_img = opt.seq_per_img
        self.clip_reward = opt.clip_reward
        self.tau_sent = opt.tau_sent

    def get_scores(self, preds, target):
        scores = []
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds.data)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        for h, r in zip(hypo, refs):
            # print('hyp:', h, "ref:", r)
            scores.append(sentence_bleu(h, r,
                                        order=self.bleu_order))
        # scale scores:
        scores = np.array(scores)
        rstats = {"r%s_raw_mean" % self.version: np.mean(scores),
                  "r%s_raw_std" % self.version: np.std(scores)}

        scores = np.clip(scores, 0, self.clip_reward) - self.clip_reward
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            print('All scores == 0')
            scores += 1
        rstats["r%s_mean" % self.version] = np.mean(scores)
        rstats["r%s_std" % self.version] = np.std(scores)
        return scores, rstats



