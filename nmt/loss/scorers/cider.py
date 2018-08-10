"""
Cider reward scorer for importance sampling
"""
import pickle
import numpy as np
from ..utils import decode_sequence
from .cider_scorer import CiderScorer


class CiderRewardScorer(object):
    """
    Evaluate CIDEr scores of given sentences wrt gt
    TODO : write scorer without decoding using only indices
    """

    def __init__(self, opt, vocab):
        self.vocab = vocab
        self.seq_per_img = opt.seq_per_img
        self.clip_reward = opt.clip_reward
        self.tau_sent = opt.tau_sent
        doc_frequency = pickle.load(open(opt.cider_df, 'rb'),
                                    encoding="iso-8859-1")
        if isinstance(doc_frequency, dict):
            self.doc_frequency = doc_frequency['freq']
            self.doc_frequency_len = doc_frequency['length']
        else:
            self.doc_frequency = doc_frequency
            self.doc_frequency_len = 40504
        self.version = 'CIDEr (tau=%.2f)' % self.tau_sent

    def get_scores(self, preds, target):
        # The reward loss:
        cider_scorer = CiderScorer(n=4, sigma=6)
        # Go to sentence space to compute scores:
        # FIXME test if numpy, variable or tensor
        hypo = decode_sequence(self.vocab, preds.data)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        for e, h in enumerate(hypo):
            ix_start = e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            cider_scorer += (h, refs[ix_start: ix_end])
            # print('hypo:', h)
            # print('refs:', refs[ix_start: ix_end])

        (score, scores) = cider_scorer.compute_score(df_mode=self.doc_frequency,
                                                     df_len=self.doc_frequency_len)
        # scale scores:
        scores = np.array(scores)
        rstats = {"rcider_raw_mean": np.mean(scores),
                  "rcider_raw_std": np.std(scores)}
        scores = np.clip(scores, 0, self.clip_reward) - self.clip_reward
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            print('All scores == 0')
            scores += 1
        rstats["rcider_mean"] = np.mean(scores)
        rstats['rcider_std'] = np.std(scores)
        return scores, rstats



