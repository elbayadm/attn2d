"""
Sampling from the model itself
"""
import numpy as np
import torch
from torch.autograd import Variable
from ..utils import to_contiguous



class GreedySampler(object):
    """
    sampling from p_\theta greedily
    """
    def __init__(self):
        self.version = 'greedy p_theta'

    def sample(self, logp, labels):
        # greedy decoding q=p_\theta
        batch_size = logp.size(0)
        seq_length = logp.size(1)
        vocab_size = logp.size(2)
        # TODO add sampling instead of argmax
        sampled = torch.max(logp, dim=2)[1].squeeze().cpu().data.numpy()
        # get p_\theta(\tilde y| x, y*)
        # Flatten
        sampled_flat = sampled.reshape((-1, 1))
        logp_sampled_greedy = to_contiguous(logp).view(-1, vocab_size).cpu().data.numpy()
        logp_sampled_greedy = np.take(logp_sampled_greedy, sampled_flat)
        logp_sampled_greedy = logp_sampled_greedy.reshape(batch_size, seq_length).mean(axis=1)
        cand_probs = np.exp(logp_sampled_greedy)
        stats = {"qpt_mean": np.mean(logp_sampled_greedy),
                 "qpt_std": np.std(logp_sampled_greedy)}

        sampled = np.hstack((np.zeros((batch_size, 1)), sampled))  # pad <BOS>
        sampled = Variable(torch.from_numpy(sampled)).cuda().type_as(labels)
        return sampled, cand_probs, stats



