"""
Samplers for data augmentation
"""

from .greedy import GreedySampler
from .hamming import HammingSampler


def init_sampler(select, opt, loader):
    """
    Wrapper for sampler selection
    """
    if select == 'greedy':
        return GreedySampler()
    elif select == 'hamming':
        return HammingSampler(opt, loader)
    else:
        raise ValueError('Unknonw sampler %s' % select)



