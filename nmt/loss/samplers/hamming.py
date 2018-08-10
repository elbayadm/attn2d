"""
Stratified sampler for the reward associated to
the negative hamming distance
"""
import math
import numpy as np
from scipy.misc import comb

import torch
from torch.autograd import Variable


def hamming_distrib_soft(m, v, tau):
    x = [np.log(comb(m, d, exact=False)) + d * np.log(v) -
         d/tau * np.log(v) - d/tau for d in range(m + 1)]
    x = np.array(x)
    p = np.exp(x)
    p /= np.sum(p)
    return p


def hamming_distrib(m, v, tau):
    try:
        x = [comb(m, d, exact=False) * (v-1)**d * math.exp(-d/tau) for d in range(m+1)]
    except:
        x = [comb(m, d, exact=False) * ((v-1) * math.exp(-1/tau))**d for d in range(m+1)]
    x = np.absolute(np.array(x))  # FIXME negative values occuring !!
    Z = np.sum(x)
    return x/Z, Z


def hamming_Z(m, v, tau):
    pd = hamming_distrib(m, v, tau)
    popul = v ** m
    Z = np.sum(pd * popul * np.exp(-np.arange(m+1)/tau))
    return np.clip(Z, a_max=1e30, a_min=1)


class HammingSampler(object):
    """
    Sample a hamming distance and alter the truth
    """
    def __init__(self, opt, loader):
        self.limited = opt.limited_vocab_sub
        self.seq_per_img = opt.seq_per_img
        self.vocab_size = opt.vocab_size
        self.bos = loader.bos
        self.eos = loader.eos
        if opt.stratify_reward:
            # sampler = r
            self.tau = opt.tau_sent
            self.prefix = 'rhamm'
        else:
            # sampler = q
            self.tau = opt.tau_sent_q
            self.prefix = 'qhamm'
        self.version = 'Hamming (Vpool=%d, tau=%.2f)' % (self.limited, self.tau)

    def nmt_sample(self, logp, labels):
        # lables tokens + EOS + padding
        # ipreds : BOS + tokens + padding
        # opreds : tokens + EOS + padding
        batch_size = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        if self.limited:
            batch_vocab = np.delete(np.unique(refs), 0)
            lv = len(batch_vocab)
        else:
            lv = self.vocab_size
        distrib, Z = hamming_distrib(seq_length, lv, self.tau)
        # Sample a distance i.e. a reward
        select = np.random.choice(a=np.arange(seq_length + 1),
                                  p=distrib)
        # score = math.exp(-select / self.tau)
        # score = distrib[select]
        score = math.exp(-select / self.tau) / Z
        stats = {"sent_mean": score,
                 "sent_std": 0}

        # Format preds by changing d=select tokens at random
        preds = refs
        # choose tokens to replace
        change_index = np.random.randint(seq_length, size=(batch_size, select))
        rows = np.arange(batch_size).reshape(-1, 1).repeat(select, axis=1)
        # select substitutes
        if self.limited:
            select_index = np.random.choice(batch_vocab, size=(batch_size, select))
        else:
            select_index = np.random.randint(low=4,
                                             high=self.vocab_size,
                                             size=(batch_size, select))
        # print("Selected:", select_index)
        preds[rows, change_index] = select_index
        # print('opreds:', preds)
        ipreds = preds[:, :-1]
        ipreds[ipreds == self.eos] = 0  # sub with PAD
        ipreds_matrix = np.hstack((np.ones((batch_size, 1)) * self.bos, ipreds))  # padd <BOS>
        # print('ipreds:', ipreds_matrix)
        ipreds_matrix = Variable(torch.from_numpy(ipreds_matrix)).cuda().type_as(labels)
        opreds_matrix = Variable(torch.from_numpy(preds)).cuda().type_as(labels)
        return ipreds_matrix, opreds_matrix, np.ones(batch_size) * score, stats



