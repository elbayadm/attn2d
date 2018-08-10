"""
Samplers for data augmentation
"""

from .greedy import GreedySampler
from .hamming import HammingSampler
from .ngram import NgramSampler


def init_sampler(select, opt, loader):
    """
    Wrapper for sampler selection
    """
    if select == 'greedy':
        return GreedySampler()
    elif select == 'hamming':
        return HammingSampler(opt, loader)
    elif select == 'ngram':
        return NgramSampler(opt)
    else:
        raise ValueError('Unknonw sampler %s' % select)



