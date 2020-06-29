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


def keeptopk(v, k):
    """
    Takes 3-dim input
    """
    topk, topkind = v.topk(k, dim=-1)
    mask = (v > topk[..., k-1:].expand_as(v)).type_as(v)  # 1 if topk
    v = v * mask
    return v


def keeptopk_masked(v, k, inf=1e6):
    topk, topkind = v.topk(k, dim=-1)
    mask = (v < topk[..., k-1:].expand_as(v)).type_as(v)
    v = v * (1 - mask) + (-inf * mask)
    return v



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
    quantized vectors are only representatives of classes
    Each class is associated with a weight-matrix
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
        # The weights book
        self.pw_w = Parameter(torch.Tensor(self.ne,
                                           self.output_dim *
                                           self.input_dim))
        # encode the input to be quantized
        self.map = Linear(self.key_dim,
                          self.reduce_dim)
        # Code book
        # self.register_buffer('centroids',
                             # torch.Tensor(self.ne, self.reduce_dim))
        self.centroids = Parameter(torch.Tensor(self.ne, self.reduce_dim))
        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w, a=-a, b=a)
            nn.init.normal_(self.centroids, mean=0, std=args.init_centroids)

        # Beta:
        self.commit = args.commitment_scale
        self.imp_loss_scale = 30
        self.kmeans_scale = args.train_embeddings_scale
        # lambda
        self.decay = args.ema_decay
        # distance between the points and the centroids
        self.distance = args.distance
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        # Counts for EMA:
        # self.register_buffer('counts', torch.zeros(self.ne))
        
    def assign(self, points, distance='euclid', greedy=False):
        # points = points.data
        centroids = self.centroids
        if distance == 'cosine':
            # nearest neigbor in the centroids (cosine distance):
            points = F.normalize(points, dim=-1)
            centroids = F.normalize(centroids, dim=-1)

        distances = (torch.sum(points**2, dim=1, keepdim=True) +
                     torch.sum(centroids**2, dim=1, keepdim=True).t() -
                     2 * torch.matmul(points, centroids.t()))  # T*B, e
        print('Distances:', distances[:3])
        if not greedy:
            logits = - distances
            resp = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
        else:
            # Greedy non-differentiable responsabilities:
            indices = torch.argmin(distances, dim=-1)  # T*B
            resp = torch.zeros(points.size(0), self.ne).type_as(points)
            resp.scatter_(1, indices.unsqueeze(1), 1)
        return resp

    def forward(self, x, key):
        T, B, C = x.size()
        if key is None:
            key = x.contiguous().view(T*B, C)
            Tr = T
            x0 = x
        else:
            print('Input key:', key.size())
            x0 = key
            key = key.view(B, -1)
            Tr = 1
        key = self.map(key)  # reduce_dim
        # self.centroids = self.centroids.type_as(x).to(device=x.device)
        # self.experts = torch.arange(self.ne, dtype=torch.long, device=x.device)
        if self.training:
            # Assign and update
            for s in range(1):
                # self.centroids = F.dropout(self.centroids, p=0.3)
                # cluster:
                resp = self.assign(
                    key, self.distance
                )
                batch_counts = resp.sum(dim=0).flatten().clone().data
                print('Step %d' % s, batch_counts)
                # update the counts:
                # for i in self.experts:
                    # self.counts[i] = self.decay * self.counts[i] + \
                            # (1 - self.decay) * batch_counts[i]
                    # # update the centroids:
                    # mean_assigned = torch.sum(
                        # resp[:, i:i+1] * key, dim=0
                    # ) / batch_counts[i]
                    # self.centroids[i] = self.decay * self.centroids[i] + \
                            # (1 - self.decay) * mean_assigned.clone().data
        else:
            # assign with greedy sampling
            resp = self.assign(key, self.distance, greedy=True)

        quantized = torch.matmul(resp, self.centroids)  # z_q(x) T*B, C
        # push the keys to commit
        commitment_loss = torch.mean((key - quantized.detach())**2)
        kmeans_loss = torch.mean((key.detach() - quantized)**2)

        # stats:
        stdc = self.centroids.std(dim=0).mean()
        stdq = quantized.std(dim=0).mean()
        stdk = key.std(dim=0).mean()
        stdi = x.contiguous().view(T*B, -1).std(dim=0).mean()
        print('|| std(centroids): %.3e\n|| std(quantized): %.3e\n|| std(keys): %.3e\n|| std(input): %.3e' % (stdc, stdq, stdk, stdi))

        importance = resp.sum(dim=0)
        imp_loss = torch.std(importance) / torch.mean(importance)

        loss = self.commit * commitment_loss + self.imp_loss_scale * imp_loss + self.kmeans_scale * kmeans_loss
        w = torch.matmul(resp, self.pw_w).view(Tr, B, self.output_dim, self.input_dim)
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss


class ConditionalDLFactorized16(nn.Module):
    """
    Quantized Dynamic layer with EMA
    quantized vectors are only representatives of classes
    Each class is associated with a weight-matrix
    The responsabilites are sampled via gumbel-softmax during training
    GMM instad of K-means with variance tau*I
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
        # The weights book
        self.pw_w = Parameter(torch.Tensor(self.ne,
                                           self.output_dim *
                                           self.input_dim))
        # encode the input to be quantized
        self.map = Linear(self.key_dim,
                          self.reduce_dim)
        # Code book
        self.register_buffer('centroids',
                             torch.Tensor(self.ne, self.reduce_dim))
        self.register_buffer('prior',
                             torch.ones(1, self.ne) / self.ne)

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w, a=-a, b=a)
            nn.init.normal_(self.centroids, mean=0, std=args.init_centroids)

        # Beta:
        self.commit = args.commitment_scale
        # lambda
        self.decay = args.ema_decay
        # distance between the points and the centroids
        self.use_gumbel = args.use_gumbel
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard

        # Counts for EMA:
        self.register_buffer('counts', torch.zeros(self.ne))
        
    def assign(self, points, distance='euclid', greedy=False):
        centroids = F.dropout(self.centroids, p=0.3)
        if distance == 'cosine':
            # nearest neigbor in the centroids (cosine distance):
            points = F.normalize(points, dim=-1)
            centroids = F.normalize(centroids, dim=-1)

        distances = (torch.sum(points**2, dim=1, keepdim=True) +
                     torch.sum(centroids**2, dim=1, keepdim=True).t() -
                     2 * torch.matmul(points, centroids.t()))  # T*B, e
        print('Distances:', distances[:3])
        if not greedy:
            resp = - .5 * self.tau * distances - self.reduce_dim / 2 * math.log(2 * math.pi * self.tau) + torch.log(self.prior)
        else:
            # Greedy non-differentiable responsabilities:
            indices = torch.argmin(distances, dim=-1)  # T*B
            resp = torch.zeros(points.size(0), self.ne).type_as(points)
            resp.scatter_(1, indices.unsqueeze(1), 1)
        return resp

    def forward(self, x, key):
        T, B, C = x.size()
        x0 = x
        key = self.map(
            x.contiguous().view(T*B, C)
        )  # reduce_dim
        self.centroids = self.centroids.type_as(x).to(device=x.device)
        self.prior = self.prior.type_as(x).to(device=x.device)
        self.experts = torch.arange(self.ne, dtype=torch.long, device=x.device)
        if self.training:
            # Assign and update
            for s in range(1):
                logresp = self.assign(key)
                print('Resp:', logresp)
                denom = torch.logsumexp(logresp, dim=-1, keepdim=True)
                # print('denom', denom)
                resp_norm = torch.exp(logresp - denom)  # gamma
                # print('Resp norm', resp_norm)
                batch_counts = resp_norm.sum(dim=0).flatten().clone().data  # Nk
                print('Step %d' % s, batch_counts)
                # update the counts:
                for i in self.experts:
                    self.counts[i] = self.decay * self.counts[i] + \
                            (1 - self.decay) * batch_counts[i]
                    # update the centroids:
                    mean_assigned = torch.sum(
                        resp_norm[:, i:i+1] * key, dim=0
                    ).clone().data / batch_counts[i]
                    self.centroids[i] = self.decay * self.centroids[i] + \
                            (1 - self.decay) * mean_assigned
                    # update the prior:
                    self.prior = self.counts / self.counts.sum(dim=-1, keepdim=True)
                nll = - denom.sum()
                # print('nll:', nll)
                resp = resp_norm
        else:
            # assign with greedy sampling
            resp = self.assign(key, greedy=True)
            nll = - torch.log(resp.sum(dim=-1)).sum()
        # stats:
        stdc = self.centroids.std(dim=0).mean()
        stdq = resp.std(dim=0).mean()
        stdk = key.std(dim=0).mean()
        stdi = x.contiguous().view(T*B, -1).std(dim=0).mean()
        print('|| std(centroids): %.3e\n|| std(resp): %.3e\n|| std(keys): %.3e\n|| std(input): %.3e' % (stdc, stdq, stdk, stdi))
        loss = self.commit * nll
        w = torch.matmul(resp, self.pw_w).view(T, B, self.output_dim, self.input_dim)
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(x0)
        return x, loss


class ConditionalDLFactorized16_(nn.Module):
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
        self.use_gumbel = args.use_gumbel
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
    Mixture of experts with gumbel-softmax and the importance loss
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(self.ne, out_features * in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            # outsider influence:
            scales = torch.softmax(self.pw_scales(key), dim=-1).unsqueeze(-1).unsqueeze(-1)  # 1, B, ne, 1, 1
            w = self.pw_w1 * scales  # 1, B, ne, C_out, C_in
            w = torch.sum(w, dim=2)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(key)
            return x, loss
        else:
            x0 = x
            if self.training:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

            else:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

                # For the new exp with ne600 eval is soft as well
                # logits = self.assign(x.contiguous().view(T*B, C))
                # indices = torch.argmax(logits, dim=-1)
                # resp = torch.zeros(logits.size(0), self.ne).type_as(logits)
                # resp.scatter_(1, indices.unsqueeze(1), 1)
            importance = resp.sum(dim=0) 
            loss = self.loss_scale * torch.std(importance) / torch.mean(importance)

            print('importance', importance.data.round())
            w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
            w = w.view(T, B, self.output_dim, self.input_dim)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(x0)
            return x, loss


class ConditionalDLFactorized17_(nn.Module):
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


class ConditionalDLFactorized18_(nn.Module):
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


class ConditionalDLFactorized18(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    + Top-k gating
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.topk = args.dynamic_topk

        self.assign = Linear(self.key_dim, self.ne)
        self.noise = Linear(self.key_dim, self.ne, bias=False)

        self.pw_w1 = Parameter(torch.Tensor(self.ne, out_features * in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        
    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            # outsider influence:
            scales = torch.softmax(self.pw_scales(key), dim=-1).unsqueeze(-1).unsqueeze(-1)  # 1, B, ne, 1, 1
            w = self.pw_w1 * scales  # 1, B, ne, C_out, C_in
            w = torch.sum(w, dim=2)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(key)
            return x, loss
        else:
            x0 = x
            key = x.contiguous().view(T*B, C)
            energies = self.assign(key)
            
            if self.training:
                noise = F.softplus(self.noise(key)) * torch.randn_like(energies)
                energies = keeptopk_masked(energies + noise, self.topk)
                if self.tau:
                    resp = F.gumbel_softmax(
                        energies,
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = F.softmax(
                        energies,
                        dim=-1
                        )  # T*B, ne

            else:
                energies = keeptopk_masked(energies, self.topk)
                if self.tau:
                    resp = F.gumbel_softmax(
                        energies,
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = F.softmax(
                        energies,
                        dim=-1
                        )  # T*B, ne

                # indices = torch.argmax(energies, dim=-1)
                # resp = torch.zeros(energies.size(0), self.ne).type_as(energies)
                # resp.scatter_(1, indices.unsqueeze(1), 1)
            importance = resp.sum(dim=0)
            loss = self.loss_scale * torch.std(importance) / torch.mean(importance)

            print('importance', importance.data.round())
            w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
            w = w.view(T, B, self.output_dim, self.input_dim)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(x0)
            return x, loss


class ConditionalDLFactorized19(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    Surrounded by a residual connection
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        assert self.input_dim == self.output_dim
        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(self.ne, out_features * in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            Tr = 1
        else:
            key = x
            Tr = T

        if self.tau:
            resp = F.gumbel_softmax(
                self.assign(key.contiguous().view(Tr*B, self.key_dim)),
                tau=self.tau,
                hard=self.hard
            )  # T*B, ne
        else:
            resp = torch.softmax(
                self.assign(key.contiguous().view(Tr*B, self.key_dim)),
                dim=-1
            )  # T*B, ne
    
        # Old evaluation, for the new exp with ne600 eval is soft as well
        # logits = self.assign(x.contiguous().view(T*B, C))
        # indices = torch.argmax(logits, dim=-1)
        # resp = torch.zeros(logits.size(0), self.ne).type_as(logits)
        # resp.scatter_(1, indices.unsqueeze(1), 1)
        importance = resp.sum(dim=0) 
        loss = self.loss_scale * torch.std(importance) / torch.mean(importance)
        print('importance', importance.data.round())
        residual = x
        w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
        w = w.view(Tr, B, self.output_dim, self.input_dim)
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        if self.pw_bias is not None:
            x = x + self.pw_bias(key)
        return x + residual, loss


class ConditionalDLFactorized20(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    Resiudal connection + a non-linearity
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        assert self.input_dim == self.output_dim

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(self.ne, out_features * in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            # outsider influence:
            scales = torch.softmax(self.pw_scales(key), dim=-1).unsqueeze(-1).unsqueeze(-1)  # 1, B, ne, 1, 1
            w = self.pw_w1 * scales  # 1, B, ne, C_out, C_in
            w = torch.sum(w, dim=2)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(key)
            return x, loss
        else:
            residual = x
            if self.training:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

            else:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

                # For the new exp with ne600 eval is soft as well
                # logits = self.assign(x.contiguous().view(T*B, C))
                # indices = torch.argmax(logits, dim=-1)
                # resp = torch.zeros(logits.size(0), self.ne).type_as(logits)
                # resp.scatter_(1, indices.unsqueeze(1), 1)
            importance = resp.sum(dim=0)
            loss = self.loss_scale * torch.std(importance) / torch.mean(importance)

            print('importance', importance.data.round())
            w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
            w = w.view(T, B, self.output_dim, self.input_dim)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(residual)
            x = torch.sigmoid(x) + residual
            return x, loss


class ConditionalDLFactorized21(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    Resiudal connection + a non-linearity
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        assert self.input_dim == self.output_dim

        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(self.ne, out_features * in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            # outsider influence:
            scales = torch.softmax(self.pw_scales(key), dim=-1).unsqueeze(-1).unsqueeze(-1)  # 1, B, ne, 1, 1
            w = self.pw_w1 * scales  # 1, B, ne, C_out, C_in
            w = torch.sum(w, dim=2)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(key)
            return x, loss
        else:
            residual = x
            if self.training:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

            else:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

                # For the new exp with ne600 eval is soft as well
                # logits = self.assign(x.contiguous().view(T*B, C))
                # indices = torch.argmax(logits, dim=-1)
                # resp = torch.zeros(logits.size(0), self.ne).type_as(logits)
                # resp.scatter_(1, indices.unsqueeze(1), 1)
            importance = resp.sum(dim=0)
            loss = self.loss_scale * torch.std(importance) / torch.mean(importance)

            print('importance', importance.data.round())
            w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
            w = w.view(T, B, self.output_dim, self.input_dim)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            x = x + residual  # v1 sigmoid on x before residual
            if self.pw_bias is not None:
                x = x + self.pw_bias(residual)
            return x, loss


class ConditionalDLFactorized22(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    + Apply then gate
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(1, 1, self.ne, out_features, in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            # outsider influence:
            scales = torch.softmax(self.pw_scales(key), dim=-1).unsqueeze(-1).unsqueeze(-1)  # 1, B, ne, 1, 1
            w = self.pw_w1 * scales  # 1, B, ne, C_out, C_in
            w = torch.sum(w, dim=2)
            x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            if self.pw_bias is not None:
                x = x + self.pw_bias(key)
            return x, loss
        else:
            x0 = x
            if self.training:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

            else:
                if self.tau:
                    resp = F.gumbel_softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        tau=self.tau,
                        hard=self.hard
                    )  # T*B, ne
                else:
                    resp = torch.softmax(
                        self.assign(x.contiguous().view(T*B, C)),
                        dim=-1
                    )  # T*B, ne

                # For the new exp with ne600 eval is soft as well
                # logits = self.assign(x.contiguous().view(T*B, C))
                # indices = torch.argmax(logits, dim=-1)
                # resp = torch.zeros(logits.size(0), self.ne).type_as(logits)
                # resp.scatter_(1, indices.unsqueeze(1), 1)
            importance = resp.sum(dim=0) 
            loss = self.loss_scale * torch.std(importance) / torch.mean(importance)

            print('importance', importance.data.round())
            # w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
            # w = w.view(T, B, self.output_dim, self.input_dim)
            # x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
            # if self.pw_bias is not None:
                # x = x + self.pw_bias(x0)
            # First evaluate each expert output
            resp = resp.view(T, B, self.ne, 1)
            x = torch.matmul(self.pw_w1, x.unsqueeze(2).unsqueeze(-1)).squeeze(-1)  # T, B, ne, out
            x = F.relu(x)
            x = torch.sum(resp * x, dim=2)
            if self.pw_bias is not None:
                x = x + self.pw_bias(x0)
            return x, loss


class ConditionalDLFactorized23(nn.Module):
    """
    Mixture of experts with gumbel-softmax and the importance loss
    + Apply then gate  + Residual connection
    """
    def __init__(self, in_features, out_features, key_features, args=None):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        if key_features is not None:
            self.key_dim = key_features
        else:
            self.key_dim = in_features

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
        self.assign = Linear(self.key_dim, self.ne)
        self.pw_w1 = Parameter(torch.Tensor(1, 1, self.ne, out_features, in_features))
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
        self.loss_scale = args.commitment_scale

        if args.dynamic_init_method == 1:  # default
            a = math.sqrt(6) / (in_features + out_features)
            nn.init.uniform_(self.pw_w1, a=-a, b=a)
        elif args.dynamic_init_method == 'embed':  # For output embedding
            print('Special init for embeddings')
            assert bias == 'none'
            nn.init.normal_(self.pw_w1, mean=0, std=self.input_dim ** -0.5)

    def forward(self, x, key):
        T, B, C = x.size()
        loss = torch.zeros(1).type_as(x).to(x.device)
        if key is not None:
            Tr = 1
        else:
            key = x
            Tr = T

        if self.tau:
            resp = F.gumbel_softmax(
                self.assign(key.contiguous().view(Tr*B, self.key_dim)),
                tau=self.tau,
                hard=self.hard
            )  # T*B, ne
        else:
            resp = torch.softmax(
                self.assign(key.contiguous().view(Tr*B, self.key_dim)),
                dim=-1
            )  # T*B, ne

        importance = resp.sum(dim=0)
        loss = self.loss_scale * torch.std(importance) / torch.mean(importance)
        print('importance', importance.data.round())
        # w = torch.matmul(resp, self.pw_w1)  # T*B, C_out * C_in
        # w = w.view(T, B, self.output_dim, self.input_dim)
        # x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1)
        # if self.pw_bias is not None:
            # x = x + self.pw_bias(x0)
        # First evaluate each expert output
        resp = resp.view(Tr, B, self.ne, 1)
        residual = x
        x = torch.matmul(self.pw_w1, x.unsqueeze(2).unsqueeze(-1)).squeeze(-1)  # T, B, ne, out
        x = F.relu(x)
        x = torch.sum(resp * x, dim=2)
        if self.pw_bias is not None:
            x = x + self.pw_bias(key)
        return x + residual, loss



