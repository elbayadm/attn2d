# Copyright (c) 2017-present, Facebook, Inc.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions.normal import Normal
from .quantizer import VectorQuantizer
from .multihead_attention import MultiheadAttention


def saturated_sigmoid(x):
    return torch.max(0, torch.min(1, 1.2 * torch.sigmoid(x) - 0.1))


def bit_to_int(x, base=2):
    num_bits = x.size(-1)
    x = x.int()
    decimal = torch.sum(torch.stack([x[..., i] * base ** i for i in range(num_bits)], dim=-1), dim=-1)
    return  decimal


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class ConditionalDLFactorized15(nn.Module):
    """
    Quantized Dynamic layer with EMA
    quantized vectors are used as the U matrix in a UV factorization
    The responsabilites are sampled via gumbel-softmax during training
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.key_dim = key_features

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

        self.reduce_dim = args.reduce_dimension
        bias = args.learnable_bias
        if bias == 'dynamic':
            self.pw_bias = nn.Linear(self.key_dim, out_features, bias=True)
            nn.init.constant_(self.pw_bias.bias, 0.)
            nn.init.constant_(self.pw_bias.weight, 0.)
        elif bias == 'constant':
            self.pw_B = Parameter(torch.zeros(1, 1, out_features))
            nn.init.constant_(self.pw_B, 0.)
            self.pw_bias = lambda x: self.pw_B
        elif bias == 'none':
            self.pw_bias = None
        else:
            raise ValueError('unknown bias mode')

        self.ne = args.dynamic_nexperts
        # The V matrix
        self.pw_w1 = Parameter(torch.Tensor(1, 1,
                                            self.reduce_dim,
                                            self.input_dim))
        # encode the input to be quantized > The U matrix
        self.map = Linear(self.key_dim,
                          self.reduce_dim * self.output_dim)
        # Code book
        self.register_buffer('centroids',
                             torch.Tensor(self.ne, self.reduce_dim * self.output_dim))

        if args.dynamic_init_method == 1:  # default
            nn.init.xavier_uniform_(self.pw_w1)
            nn.init.normal_(self.centroids, mean=0, std=args.init_centroids)

        # Beta:
        self.commit = args.commitment_scale
        # lambda
        self.decay = args.ema_decay
        # distance between the points and the centroids
        self.distance = args.distance
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard

        # Counts for EMA:
        self.register_buffer('counts', torch.zeros(self.ne))
        
    def assign(self, points, distance='euclid', greedy=False):
        points = points.data
        centroids = self.centroids
        if distance == 'cosine':
            # nearest neigbor in the centroids (cosine distance):
            points = F.normalize(points, dim=-1)
            centroids = F.normalize(centroids, dim=-1)

        distances = (torch.sum(points**2, dim=1, keepdim=True) +
                     torch.sum(centroids**2, dim=1, keepdim=True).t() -
                     2 * torch.matmul(points, centroids.t()))  # T*B, e
        if not greedy:
            logits = - distances
            resp = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
            # batch_counts = resp.sum(dim=0).view(-1).data

        else:
            # Greedy non-differentiable responsabilities:
            indices = torch.argmin(distances, dim=-1)  # T*B
            resp = torch.zeros(points.size(0), self.ne).type_as(points)
            resp.scatter_(1, indices.unsqueeze(1), 1)
        return resp

    def forward(self, x, key):
        T, B, C = x.size()
        x0 = x
        key = x.contiguous()
        key = key.view(T*B, C)
        key = self.map(
            key.view(T*B, C)
        )  # T*B, C  # z_e(x)
        self.centroids = self.centroids.type_as(x).to(device=x.device)
        self.experts = torch.arange(self.ne, dtype=torch.long, device=x.device)

        if self.training:
            # Assign and update
            for s in range(1):
                # cluster:
                resp = self.assign(
                    key, self.distance
                )
                batch_counts = resp.sum(dim=0).flatten().clone().data
                print('Step %d' % s, batch_counts)
                # update the counts:
                for i in self.experts:
                    self.counts[i] = self.decay * self.counts[i] + \
                            (1 - self.decay) * batch_counts[i]
                    # update the centroids:
                    mean_assigned = torch.mean(
                        resp[:, i:i+1] * key, dim=0
                    ).clone().data
                    self.centroids[i] = self.decay * self.centroids[i] + \
                            (1 - self.decay) * mean_assigned
        else:
            # assign with greedy sampling
            resp = self.assign(key, self.distanc, greedy=True)

        quantized = torch.matmul(resp, self.centroids)  # z_q(x) T*B, C

        # push the keys to commit
        if self.distance == 'cosine':
            commitment_loss = torch.mean((
                F.normalize(key, dim=-1, p=2) - 
                F.normalize(quantized.detach(), dim=-1, p=2)
            )**2)
        elif self.distance == 'euclid':
            commitment_loss = torch.mean((key - quantized.detach())**2)
        # stats:
        stdc = self.centroids.std(dim=0).mean()
        stdq = quantized.std(dim=0).mean()
        stdk = key.std(dim=0).mean()
        print('|| std(centroids): %.3e\n|| std(quantized): %.3e\n|| std(keys): %.3e' % (stdc, stdq, stdk))
        loss = self.commit * commitment_loss
        w = quantized.view(T, B, self.reduce_dim, self.input_dim)
        w = torch.matmul(w, self.pw_w1)  # 1, B, C_out, C_in
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss


class ConditionalDLFactorized16(nn.Module):
    """
    Quantized Dynamic layer with EMA
    quantized vectors are used as the U matrix in a UV factorization
    The responsabilites are sampled via gumbel-softmax during training
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.key_dim = key_features

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

        self.reduce_dim = args.reduce_dimension
        bias = args.learnable_bias
        if bias == 'dynamic':
            self.pw_bias = nn.Linear(self.key_dim, out_features, bias=True)
            nn.init.constant_(self.pw_bias.bias, 0.)
            nn.init.constant_(self.pw_bias.weight, 0.)
        elif bias == 'constant':
            self.pw_B = Parameter(torch.zeros(1, 1, out_features))
            nn.init.constant_(self.pw_B, 0.)
            self.pw_bias = lambda x: self.pw_B
        elif bias == 'none':
            self.pw_bias = None
        else:
            raise ValueError('unknown bias mode')

        self.ne = args.dynamic_nexperts
        # The V matrix
        self.pw_w1 = Parameter(torch.Tensor(1, 1,
                                            self.reduce_dim,
                                            self.input_dim))
        # encode the input to be quantized > The U matrix
        self.map = Linear(self.key_dim,
                          self.reduce_dim * self.output_dim)
        # Code book
        self.centroids = Parameter(torch.Tensor(self.ne,
                                                self.reduce_dim * self.output_dim))

        if args.dynamic_init_method == 1:  # default
            nn.init.xavier_uniform_(self.pw_w1)
            nn.init.normal_(self.centroids, mean=0, std=args.init_centroids)

        # Beta:
        self.commit = args.commitment_scale
        # lambda
        self.decay = args.ema_decay
        # distance between the points and the centroids
        self.distance = args.distance
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard

        # Counts for EMA:
        self.register_buffer('counts', torch.zeros(self.ne))
        
    def assign(self, points, distance='euclid', greedy=False):
        points = points.data
        centroids = self.centroids
        if distance == 'cosine':
            # nearest neigbor in the centroids (cosine distance):
            points = F.normalize(points, dim=-1)
            centroids = F.normalize(centroids, dim=-1)

        distances = (torch.sum(points**2, dim=1, keepdim=True) +
                     torch.sum(centroids**2, dim=1, keepdim=True).t() -
                     2 * torch.matmul(points, centroids.t()))  # T*B, e
        if not greedy:
            logits = - distances
            resp = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
            # batch_counts = resp.sum(dim=0).view(-1).data

        else:
            # Greedy non-differentiable responsabilities:
            indices = torch.argmin(distances, dim=-1)  # T*B
            resp = torch.zeros(points.size(0), self.ne).type_as(points)
            resp.scatter_(1, indices.unsqueeze(1), 1)
        return resp

    def forward(self, x, key):
        T, B, C = x.size()
        x0 = x
        key = x.contiguous()
        key = key.view(T*B, C)
        key = self.map(
            key.view(T*B, C)
        )  # T*B, C  # z_e(x)
        # self.centroids = self.centroids.type_as(x).to(device=x.device)
        self.experts = torch.arange(self.ne, dtype=torch.long, device=x.device)

        if self.training:
            # Assign and update
            for s in range(1):
                # cluster:
                resp = self.assign(
                    key, self.distance
                )
                batch_counts = resp.sum(dim=0).flatten().clone().data
                print('Step %d' % s, batch_counts)
        else:
            # assign with greedy sampling
            resp = self.assign(key, self.distance, greedy=True)

        quantized = torch.matmul(resp, self.centroids)  # z_q(x) T*B, C

        # push the keys to commit
        if self.distance == 'cosine':
            commitment_loss = torch.mean((
                F.normalize(key, dim=-1, p=2) - 
                F.normalize(quantized.detach(), dim=-1, p=2)
            )**2)
        elif self.distance == 'euclid':
            commitment_loss = torch.mean((key - quantized.detach())**2)
        # stats:
        stdc = self.centroids.std(dim=0).mean()
        stdq = quantized.std(dim=0).mean()
        stdk = key.std(dim=0).mean()
        print('|| std(centroids): %.3e\n|| std(quantized): %.3e\n|| std(keys): %.3e' % (stdc, stdq, stdk))
        loss = self.commit * commitment_loss
        w = quantized.view(T, B, self.reduce_dim, self.input_dim)
        w = torch.matmul(w, self.pw_w1)  # 1, B, C_out, C_in
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss


class ConditionalDLFactorized17(nn.Module):
    """
    Quantized Dynamic layer with EMA
    quantized vectors are used as the U matrix in a UV factorization
    The responsabilites are sampled via gumbel-softmax during training
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.key_dim = key_features

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

        self.reduce_dim = args.reduce_dimension
        bias = args.learnable_bias
        if bias == 'dynamic':
            self.pw_bias = nn.Linear(self.key_dim, out_features, bias=True)
            nn.init.constant_(self.pw_bias.bias, 0.)
            nn.init.constant_(self.pw_bias.weight, 0.)
        elif bias == 'constant':
            self.pw_B = Parameter(torch.zeros(1, 1, out_features))
            nn.init.constant_(self.pw_B, 0.)
            self.pw_bias = lambda x: self.pw_B
        elif bias == 'none':
            self.pw_bias = None
        else:
            raise ValueError('unknown bias mode')

        self.ne = args.dynamic_nexperts
        # The V matrix
        self.pw_w1 = Parameter(torch.Tensor(1, 1,
                                            self.reduce_dim,
                                            self.input_dim))
        # encode the input to be quantized > The U matrix
        self.map = Linear(self.key_dim,
                          self.reduce_dim * self.output_dim)
        # Code book
        self.centroids = Parameter(torch.Tensor(self.ne,
                                                self.reduce_dim * self.output_dim))

        if args.dynamic_init_method == 1:  # default
            nn.init.xavier_uniform_(self.pw_w1)
            nn.init.normal_(self.centroids, mean=0, std=args.init_centroids)

        # Beta:
        self.commit = args.commitment_scale
        # lambda
        self.decay = args.ema_decay
        # distance between the points and the centroids
        self.distance = args.distance
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard

        # Counts for EMA:
        self.register_buffer('counts', torch.zeros(self.ne))
        
    def assign(self, points, distance='euclid', greedy=False):
        # points = points.data  # the only diff from 16
        centroids = self.centroids
        if distance == 'cosine':
            # nearest neigbor in the centroids (cosine distance):
            points = F.normalize(points, dim=-1)
            centroids = F.normalize(centroids, dim=-1)

        distances = (torch.sum(points**2, dim=1, keepdim=True) +
                     torch.sum(centroids**2, dim=1, keepdim=True).t() -
                     2 * torch.matmul(points, centroids.t()))  # T*B, e
        if not greedy:
            logits = - distances
            resp = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
            # batch_counts = resp.sum(dim=0).view(-1).data

        else:
            # Greedy non-differentiable responsabilities:
            indices = torch.argmin(distances, dim=-1)  # T*B
            resp = torch.zeros(points.size(0), self.ne).type_as(points)
            resp.scatter_(1, indices.unsqueeze(1), 1)
        return resp

    def forward(self, x, key):
        T, B, C = x.size()
        x0 = x
        key = x.contiguous()
        key = key.view(T*B, C)
        key = self.map(
            key.view(T*B, C)
        )  # T*B, C  # z_e(x)
        # self.centroids = self.centroids.type_as(x).to(device=x.device)
        self.experts = torch.arange(self.ne, dtype=torch.long, device=x.device)

        if self.training:
            # Assign and update
            for s in range(1):
                # cluster:
                resp = self.assign(
                    key, self.distance
                )
                batch_counts = resp.sum(dim=0).flatten().clone().data
                print('Step %d' % s, batch_counts)
        else:
            # assign with greedy sampling
            resp = self.assign(key, self.distance, greedy=True)

        quantized = torch.matmul(resp, self.centroids)  # z_q(x) T*B, C

        # push the keys to commit
        if self.distance == 'cosine':
            commitment_loss = torch.mean((
                F.normalize(key, dim=-1, p=2) - 
                F.normalize(quantized.detach(), dim=-1, p=2)
            )**2)
        elif self.distance == 'euclid':
            commitment_loss = torch.mean((key - quantized.detach())**2)
        # stats:
        stdc = self.centroids.std(dim=0).mean()
        stdq = quantized.std(dim=0).mean()
        stdk = key.std(dim=0).mean()
        print('|| std(centroids): %.3e\n|| std(quantized): %.3e\n|| std(keys): %.3e' % (stdc, stdq, stdk))
        loss = self.commit * commitment_loss
        w = quantized.view(T, B, self.reduce_dim, self.input_dim)
        w = torch.matmul(w, self.pw_w1)  # 1, B, C_out, C_in
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss


class ConditionalDLFactorized18(nn.Module):
    """
    Quantized Dynamic layer with semantic hashing
    quantized vectors are used as the U matrix in a UV factorization
    The responsabilites are sampled via gumbel-softmax during training
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.key_dim = key_features

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

        self.reduce_dim = args.reduce_dimension
        bias = args.learnable_bias
        if bias == 'dynamic':
            self.pw_bias = nn.Linear(self.key_dim, out_features, bias=True)
            nn.init.constant_(self.pw_bias.bias, 0.)
            nn.init.constant_(self.pw_bias.weight, 0.)
        elif bias == 'constant':
            self.pw_B = Parameter(torch.zeros(1, 1, out_features))
            nn.init.constant_(self.pw_B, 0.)
            self.pw_bias = lambda x: self.pw_B
        elif bias == 'none':
            self.pw_bias = None
        else:
            raise ValueError('unknown bias mode')

        self.nbits = args.dynamic_nexperts
        self.ne = 2 ** args.dynamic_nexperts
        # The V matrix
        self.pw_w1 = Parameter(torch.Tensor(1, 1,
                                            self.reduce_dim,
                                            self.input_dim))
        # encode the input to be quantized > The U matrix
        self.map = Linear(self.key_dim, self.nbits)
        # Code book (U)
        self.pw_w21 = Parameter(torch.Tensor(self.ne,
                                             self.output_dim * self.reduce_dim))
        self.pw_w22 = Parameter(torch.Tensor(self.ne,
                                             self.output_dim * self.reduce_dim))


        if args.dynamic_init_method == 1:  # default
            nn.init.xavier_uniform_(self.pw_w1)
            nn.init.xavier_uniform_(self.pw_w21)
            nn.init.xavier_uniform_(self.pw_w22)

    def forward(self, x, key):
        loss = torch.zeros(1).type_as(x).to(x.device)
        T, B, C = x.size()
        x0 = x
        key = x.contiguous().view(T*B, C)
        key = self.map(key)
        if self.training:
            noise = torch.randn_like(key)
            key = key + noise
        z = saturated_sigmoid(key)
        bz = z.gt(0.5).type_as(z)  #+ z - z.detach()  # the gradient not the value
        if self.training:
            # mixing:
            coin = torch.rand_like(z).gt(0.5).type_as(z)
            code = coin * bz + (1-coin) * z
        else:
            code = bz

        qz1 = bit_to_int(code)
        qz2 = bit_to_int(1 - code)
        print('Part 1:', torch.unique(qz1).data.flatten())
        print('Part 2:', torch.unique(qz2).data.flatten())

        # build w
        # onehot1 = torch.zeros(T*B, self.ne).type_as(x)
        # onehot1.scatter_(1, qz1.unsqueeze(1), 1)
        # onehot2 = torch.zeros(T*B, self.ne).type_as(x)
        # onehot2.scatter_(1, qz2.unsqueeze(1), 1)
        # print("One hot:", onehot1)

        # quantized1 = torch.matmul(onehot1, self.pw_w21)  # T*B, out*reduce
        # quantized2 = torch.matmul(onehot2, self.pw_w22)
        w21 = self.pw_w21[qz1]
        w22 = self.pw_w22[qz2]

        w = (w21 + w22).view(T, B, self.output_dim, self.reduce_dim)
        w = torch.matmul(w, self.pw_w1)
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss

