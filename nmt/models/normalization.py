import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        self.alpha = nn.Parameter(self.alpha)
        self.beta = nn.Parameter(self.beta)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).expand_as(x)) / torch.sqrt(torch.var(x, 1).expand_as(x) + self.epsilon)
        if self.learnable:
            x =  self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)
