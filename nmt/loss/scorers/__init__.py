"""
Each scorer defines a get_scores function with:
    Inputs:
        - preds: the candidates xN
        - target: the references xN
    Outputs:
        - scores (N,) array of scores
        - stats: dict of statistics for traceability
"""
from .cider import CiderRewardScorer
from .bleu import BleuRewardScorer
import numpy as np


def init_scorer(method, opt, vocab):
    if method == 'constant':
        return AllIsGoodScorer()
    elif method == 'cider':
        return CiderRewardScorer(opt, vocab)
    elif 'bleu' in method:
        return BleuRewardScorer(opt, vocab)
    else:
        raise ValueError('Unknown reward %s' % method)


class AllIsGoodScorer(object):
    """
    constant scores
    """

    def __init__(self):
        self.version = "constant"

    def get_scores(self, preds, target):
        rstats = {"rconst_mean": 1,
                  "rconst_std": 0}
        return np.ones(target.size(0)), rstats



