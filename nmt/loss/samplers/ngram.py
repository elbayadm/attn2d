"""
Augmentation by altering an n-gram
"""
import random
from collections import OrderedDict
import pickle
import numpy as np
import torch
from torch.autograd import Variable


class NgramSampler(object):
    """
    Alter an n-gram
    """
    def __init__(self, opt):
        self.ngrams = pickle.load(open('data/coco-train-tok-ng-df.p', 'rb'))
        self.select_rare = opt.rare_tfidf
        self.tau = opt.tau_sent
        self.n = opt.ngram_length
        self.sub_idf = opt.sub_idf
        if self.select_rare:
            self.ngrams = OrderedDict(self.ngrams[self.n])
            freq = np.array([1/c for c in list(self.ngrams.values())])
            if self.tau:
                freq = np.exp(freq/self.tau)
            freq /= np.sum(freq)
            self.ngrams = OrderedDict({k: v for k,v in zip(list(self.ngrams), freq)})
            # print('self.ngrams:', self.ngrams)
        else:
            self.ngrams = list(self.ngrams[self.n])
        self.version = 'TFIDF, n=%d, rare=%d, tau=%.2e' % (self.n, self.select_rare, self.tau)

    def sample(self, logp, labels):
        batch_size = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        # ng = 1 + np.random.randint(4)
        ng = self.n
        stats = {"sent_mean": ng,
                 "sent_std": 0}
        # Format preds by changing d=select tokens at random
        preds = refs
        # choose an n-consecutive words to replace
        if self.sub_idf:
            # get current ngrams dfs:
            change_index = np.zeros((batch_size, 1), dtype=np.int32)
            for i in range(batch_size):
                p = np.array([self.ngrams.get(tuple(refs[i, j:j+ng]), 1)
                              for j in range(seq_length - ng)])
                p = 1/p
                p /= np.sum(p)
                change_index[i] = np.random.choice(seq_length - ng,
                                                   p=p,
                                                   size=1)
        else:
            change_index = np.random.randint(seq_length - ng, size=(batch_size, 1))
        change_index = np.hstack((change_index + k for k in range(ng)))
        rows = np.arange(batch_size).reshape(-1, 1).repeat(ng, axis=1)
        # select substitutes from occuring n-grams in the training set:
        if self.select_rare:
            picked = np.random.choice(np.arange(len(self.ngrams)),
                                      p=list(self.ngrams.values()),
                                      size=(batch_size,))
            picked_ngrams = [list(self.ngrams)[k] for k in picked]
        else:
            picked_ngrams = random.sample(self.ngrams, batch_size)
        preds[rows, change_index] = picked_ngrams
        preds_matrix = np.hstack((np.zeros((batch_size, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, np.ones(batch_size) * score, stats


