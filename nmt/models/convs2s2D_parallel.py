"""
2DConvS2S architecture
"""
import logging
import torch.nn as nn

from .convs2s2D import Convs2s2D
from .convnet import ConvNet
from .densenet import DenseNet
from .aggregator import Aggregator
from .embedding import Embedding, ConvEmbedding
from .beam_search import Beam


def _expand(tensor, dim, reps):
    # Expand 4D tensor in the source or the target dimension
    if dim == 1:
        return tensor.repeat(1, reps, 1, 1)
        # return tensor.expand(-1, reps, -1, -1)
    if dim == 2:
        return tensor.repeat(1, 1, reps, 1)
        # return tensor.expand(-1, -1, reps, -1)
    else:
        raise NotImplementedError


class Convs2s2D_Parallel(nn.DataParallel):
    def __init__(self, jobname, params,
                 src_vocab_size, trg_vocab_size, special_tokens):
        model = Convs2s2D(jobname, params, src_vocab_size, trg_vocab_size, special_tokens)
        nn.DataParallel.__init__(self, model)
        self.logger = logging.getLogger(jobname)
        self.version = 'conv'
        self.params = params
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.pad_token = special_tokens['PAD']
        # assert self.pad_token == 0, "Padding token should be 0"
        self.bos_token = special_tokens['BOS']
        self.eos_token = special_tokens['EOS']
        self.kernel_size = params['network']['kernels'][0]  # assume using the same kernel size all over

    def init_weights(self):
        self.module.init_weights()

    def _forward(self, X, src_lengths=None):
        return self.module._forward(self, X, src_lengths)

    def update(self, X, src_lengths=None):
        return self.module.update(X, src_lengths)

    def sample(self, data_src, scorer=None, kwargs={}):
        return self.module.sample(data_src, scorer, kwargs)


