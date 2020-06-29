# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def saturated_sigmoid(x):
    return torch.clamp(1.2 * torch.sigmoid(x) - 0.1, min=0, max=1)


class DynamicHalterNclasses(nn.Module):
    """
    A single halting signal predicting 1 out of the N exits
    conditionned on the first block hidden states
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()

        self.detach_before_classifier = args.detach_before_classifier
        self.num_exits = num_exits
        self.path_probability = args.path_probability

        self.drop_rate = args.halt_dropout

        if args.halt_nonlin == 'relu':
            nonlin = nn.ReLU
        elif args.halt_nonlin == 'sigmoid':
            nonlin = nn.Sigmoid
        elif args.halt_nonlin == 'tanh':
            nonlin = nn.Tanh

        self.halting_predictor = nn.Sequential()
        for n in range(args.halt_layers):
            self.halting_predictor.add_module(
                'map%d' % n, Linear(embed_dim, embed_dim)
            )
            self.halting_predictor.add_module(
                'nonlin%d' % n, nonlin()
            )

        final_map = Linear(embed_dim, num_exits)
        self.halting_predictor.add_module('map', final_map)

    def forward(self, decoder_inner_states):
        x = decoder_inner_states[0]  # predict the exit after the first block
        if self.detach_before_classifier:
            x = x.detach()
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.halting_predictor(x.transpose(0, 1))
        x = F.log_softmax(x, dim=-1)  # log(p_t^n)  B, T, N
        return x


class DynamicHalter(nn.Module):
    """
    No halting
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        
    def step(self, x, n, **kwargs):
        return None, None, None

    def forward(self, decoder_inner_states):
        return None


class DynamicHalterUniform(nn.Module):
    """
    Bernoulli  1/N, 1/N-1, ... 1/2
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits
        self.thresholds = args.thresholds
        
    def step(self, x, n, **kwargs):
        """
        n is the index of the previous block
        returns the binary decision, the halting signal and the logits
        """
        T, B, C = x.size()
        p = x.new_empty((T, B)).fill_(1 / (self.num_exits - n + 1))
        return torch.bernoulli(p).byte(), None,  None

    def forward(self, decoder_inner_states):
        """
        In training mode, returns the logits
        """
        return None


class DynamicHalterCumul(nn.Module):
    """
    Proba : cumul
        The halting signals are cumulated to get the exit probability
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits
        self.path_probability = args.path_probability

        self.detach_before_classifier = args.detach_before_classifier
        self.separate_halting_predictors = args.separate_halting_predictors
        self.shift_block_input = args.shift_block_input
        self.adapt_block_input = args.adapt_block_input
        self.thresholds = args.thresholds
        self.use_skewed_sigmoid = args.use_saturated_sigmoid
        self.skewness = args.skewness
        self.eps = 0.01

        if self.shift_block_input:
            shifts = torch.randn((args.encoder_layers - 1,
                                  self.embed_dim)) / math.sqrt(self.embed_dim)

            self.input_shifters = nn.Parameter(shifts)

        self.halting_predictors = nn.ModuleList([])
        for n in range(self.num_exits - 1):
            map = Linear(self.embed_dim, 1, bias=args.halting_bias)
            self.halting_predictors.append(map)
            if not self.separate_halting_predictors:
                break

    def step(self, x, n, cumul=None, total_computes=None, hard_decision=False):
        """
        n is the index of the previous block
        returns the binary decision, the halting signal and the logits
        """
        if self.detach_before_classifier:
            x = x.detach()
        # If adding an embedding of the total computes:
        if self.shift_block_input:
            computes_embed = F.embedding(total_computes, self.input_shifters)
            x = x + computes_embed
        x = self.halting_predictors[n if self.separate_halting_predictors else 0](x).squeeze(-1)
        if self.use_skewed_sigmoid:
            halt = F.sigmoid(self.skewness * x)  # the log-p of halting
        else:
            halt = F.sigmoid(x)  # the log-p of halting
        if hard_decision:
            decision = (cumul + halt).ge(0.99)
            return decision, halt
        return halt  # T, B

    def forward(self, decoder_inner_states):
        """
        In training mode, returns the logits
        """
        halts = []
        x = decoder_inner_states[0]
        T, B, C = x.size()
        cumul = x.new_zeros((T, B))
        total_computes = decoder_inner_states[0].new_zeros((T, B)).long()
        for n in range(self.num_exits - 1):
            halt = self.step(decoder_inner_states[n], n, cumul, total_computes=total_computes)
            total_computes = total_computes + 1
            halts.append(halt)

        halts = torch.stack(halts, dim=-1)  # T, B, N-1
        cumul = torch.cumsum(halts, dim=-1)
        ongoing = cumul.le(1 - self.eps).float()
        # find the earliest saturation
        halts = halts * ongoing
        last = 1 - halts.sum(dim=-1, keepdim=True)  # 1 - sum h_tn up to ^n-1
        halts = halts + (1 - ongoing) * last
        halts = torch.cat((halts, last), dim=-1)  # T, B, N
        return halts.transpose(0, 1)   # p_t^n


class DynamicHalterStickBreaking(nn.Module):
    """
    Proba : Poisson binomial
        N-1 halting signals as the probability of exiting after the nth block (sigmoid)
        The final exit probability is a stick-breaking probability
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits
        self.path_probability = args.path_probability

        self.detach_before_classifier = args.detach_before_classifier
        self.separate_halting_predictors = args.separate_halting_predictors
        self.shift_block_input = args.shift_block_input
        self.adapt_block_input = args.adapt_block_input
        self.thresholds = list(args.thresholds)
        while len(self.thresholds) < self.num_exits - 1:
            self.thresholds.append(0.5)
        print('Using thresholds:', self.thresholds)

        self.use_skewed_sigmoid = args.use_saturated_sigmoid
        self.skewness = args.skewness

        if self.shift_block_input:
            shifts = torch.randn((args.encoder_layers - 1,
                                  self.embed_dim)) / math.sqrt(self.embed_dim)

            self.input_shifters = nn.Parameter(shifts)

        if args.halt_nonlin == 'relu':
            nonlin = nn.ReLU
        elif args.halt_nonlin == 'sigmoid':
            nonlin = nn.Sigmoid
        elif args.halt_nonlin == 'tanh':
            nonlin = nn.Tanh

        self.halting_predictors = nn.ModuleList([])
        for n in range(self.num_exits - 1):
            if args.halt_layers:
                map = nn.Sequential()
                for n in range(args.halt_layers):
                    map.add_module(
                        'map%d' % n, Linear(embed_dim, embed_dim)
                    )
                    map.add_module(
                        'nonlin%d' % n, nonlin()
                    )
                map.add_module('map', Linear(embed_dim, 1, bias=args.halting_bias))
            else:
                map = Linear(self.embed_dim, 1, bias=args.halting_bias)
            self.halting_predictors.append(map)
            if not self.separate_halting_predictors:
                break

    def step(self, x, n, total_computes=None, hard_decision=False, **kwargs):
        """
        n is the index of the previous block
        returns the binary decision, the halting signal and the logits
        """
        if self.detach_before_classifier:
            x = x.detach()

        # If adding an embedding of the total computes:
        if self.shift_block_input:
            computes_embed = F.embedding(total_computes, self.input_shifters)
            x = x + computes_embed
        x = self.halting_predictors[n if self.separate_halting_predictors else 0](x)
        if self.use_skewed_sigmoid:
            halt = F.logsigmoid(self.skewness * x)  # the log-p of halting
            halt_logits = torch.cat((halt, halt - self.skewnees * x), dim=-1)  # log-p of halting v. computing
        else:
            halt = F.logsigmoid(x)  # the log-p of halting
            halt_logits = torch.cat((halt, halt-x), dim=-1)  # log-p of halting v. computing
        if hard_decision:
            halt = torch.exp(halt.squeeze(-1))
            return halt.ge(self.thresholds[n])
        return halt_logits  # T, B, 2
        
    def forward(self, decoder_inner_states):
        """
        In training mode, returns p_t^n from combining all the halting signals (h_t^n)
        """
        halter_outputs = []
        x = decoder_inner_states[0]
        T, B, C = x.size()
        total_computes = decoder_inner_states[0].new_zeros((T, B)).long()
        for n in range(self.num_exits - 1):
            logits = self.step(decoder_inner_states[n], n, total_computes=total_computes)
            total_computes = total_computes + 1
            halter_outputs.append(logits)
            
        # evaluate the exit proba:
        exit_logits = []
        neg_logits = x.new_zeros((T, B))
        for n in range(len(halter_outputs)):
            exit_n = halter_outputs[n][..., 0] + neg_logits
            exit_logits.append(exit_n)
            neg_logits = neg_logits + halter_outputs[n][..., 1]
        # one last from all the compute until the end:
        exit_logits.append(neg_logits)
        exit_logits = torch.stack(exit_logits, dim=-1).transpose(0, 1).contiguous()  # B, T, N
        return exit_logits  # log(p_t^n)

    def forward_and_sample(self, decoder_inner_states):
        """
        Decide similarly to the inference mode where if h_t^n > 0.5 we exit
        """
        halter_outputs = []
        x = decoder_inner_states[0]
        T, B, C = x.size()
        total_computes = decoder_inner_states[0].new_zeros((T, B)).long()
        for n in range(self.num_exits - 1):
            logits = self.step(decoder_inner_states[n], n, total_computes=total_computes)
            total_computes = total_computes + 1
            halter_outputs.append(logits)
        # evaluate the exit proba:
        exit_logits = []
        neg_logits = x.new_zeros((T, B))
        for n in range(len(halter_outputs)):
            exit_n = halter_outputs[n][..., 0] + neg_logits
            exit_logits.append(exit_n)
            neg_logits = neg_logits + halter_outputs[n][..., 1]
        # one last from all the compute until the end:
        exit_logits.append(neg_logits)
        exit_logits = torch.stack(exit_logits, dim=-1).transpose(0, 1).contiguous()  # B, T, N  log p_t^n
        argmax_exits = torch.argmax(exit_logits, dim=-1)

        # Sample an exit given only partial halting signals:
        sampled_exits = x.new_zeros((T, B)).long()
        for n in range(len(halter_outputs)):
            sampled_exits = sampled_exits + sampled_exits.eq(0).long() * halter_outputs[n][..., 0].gt(math.log(0.5)).long() * (n+1)
        sampled_exits[sampled_exits.eq(0)] = self.num_exits
        return argmax_exits, sampled_exits.transpose(0, 1) - 1


class DynamicHalterMembership(nn.Module):
    """
    Proba : Poisson binomial
        N-1 halting signals as the probability of exiting after the nth block (sigmoid)
        The final exit probability is a stick-breaking probability
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits
        self.path_probability = args.path_probability
        self.detach_before_classifier = args.detach_before_classifier
        self.separate_halting_predictors = args.separate_halting_predictors
        self.shift_block_input = args.shift_block_input
        self.adapt_block_input = args.adapt_block_input
        self.thresholds = args.thresholds
        self.use_saturated_sigmoid = args.use_saturated_sigmoid
        self.use_skewed_sigmoid = args.use_saturated_sigmoid
        self.skewness = args.skewness

        if self.shift_block_input:
            shifts = torch.randn((args.encoder_layers - 1,
                                  self.embed_dim)) / math.sqrt(self.embed_dim)

            self.input_shifters = nn.Parameter(shifts)

        self.halting_predictors = nn.ModuleList([])
        for n in range(self.num_exits):
            map = nn.Linear(self.embed_dim, 2)
            nn.init.xavier_uniform_(map.weight)
            nn.init.constant_(map.bias, args.halting_bias)
            self.halting_predictors.append(map)
            if not self.separate_halting_predictors:
                break

    def step(self, x, n, total_computes=None, hard_decision=False):
        """
        n is the index of the previous block
        returns the binary decision, the halting signal and the logits
        """
        T, B, C = x.size()
        if self.detach_before_classifier:
            x = x.detach()
        # If adding an embedding of the total computes:
        if self.shift_block_input:
            computes_embed = F.embedding(total_computes, self.input_shifters)
            x = x + computes_embed
        x = self.halting_predictors[n if self.separate_halting_predictors else 0](x)
        halt = F.log_softmax(x, dim=-1)  # T, B, 2
        if hard_decision:
            decision = halt[..., 0] .squeeze(-1).ge(math.log(0.5))  # T, B
            return decision
        return halt

    def forward(self, decoder_inner_states):
        """
        In training mode, returns the logits
        """
        exit_logits = []
        x = decoder_inner_states[0]
        T, B, C = x.size()
        total_computes = decoder_inner_states[0].new_zeros((T, B)).long()
        for n in range(self.num_exits):
            halt = self.step(decoder_inner_states[n],
                             n, total_computes=total_computes)
            total_computes = total_computes + 1
            exit_logits.append(halt)
        exit_logits = torch.stack(exit_logits, dim=2)  # T, B, N, 2
        return exit_logits.transpose(0, 1).contiguous()


class DynamicHalterGumbel(nn.Module):
    """
    
    Similar to DynamicHalter with hard logits via Gumbel-softmax
    """
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits
        self.path_probability = args.path_probability
        self.detach_before_classifier = args.detach_before_classifier
        self.separate_halting_predictors = args.separate_halting_predictors
        self.shift_block_input = args.shift_block_input
        self.adapt_block_input = args.adapt_block_input
        self.thresholds = args.thresholds
        self.gumbel_tau = args.gumbel_tau
        print('Setting up thresholds:', self.thresholds)
        self.use_saturated_sigmoid = args.use_saturated_sigmoid

        if self.shift_block_input:
            shifts = torch.randn((args.encoder_layers - 1,
                                  self.embed_dim)) / math.sqrt(self.embed_dim)

            self.input_shifters = nn.Parameter(shifts)

        if self.path_probability not in ['bernoulli', 'uniform']:
            self.halting_predictors = nn.ModuleList([])
            for n in range(self.num_exits - 1):
                map = Linear(self.embed_dim, 2, bias=args.halting_bias)
                self.halting_predictors.append(map)
                if not self.separate_halting_predictors:
                    break
            print('Halting predictors:', self.halting_predictors)

    def step(self, x, n, cumul=None, total_computes=None):
        """
        n is the index of the upcoming block, 
        Given the current activation decide whether to go in or skip/exit.
        returns the binary decision and the log-(p, 1-p)
        """
        T, B, C = x.size()
        if self.detach_before_classifier:
            x = x.detach()
        x = self.halting_predictors[n if self.separate_halting_predictors else 0](x)
        halt_logits = F.logsigmoid(x)  # the log-p of halting
        # Apply the gumbel trick
        halt = halt_logits.view(-1, 2)
        halt = F.gumbel_softmax(halt, tau=self.gumbel_tau).view(T, B, 2)
        return halt

    def forward(self, decoder_inner_states):
        exit_logits = []
        T, B, C = decoder_inner_states[0].size()
        total_computes = decoder_inner_states[0].new_zeros((T, B)).long()
        for n, x in enumerate(decoder_inner_states):
            halt = self.step(x, n,
                             total_computes=total_computes)
            exit_logits.append(halt)
        return exit_logits


class SeqDynamicHalter(nn.Module):
    def __init__(self, args, embed_dim, num_exits):
        super().__init__()
        self.num_exits = num_exits
        self.path_probability = args.path_probability
        self.detach_before_classifier = args.detach_before_classifier
        self.drop_rate = args.halt_dropout
        if args.halt_nonlin == 'relu':
            nonlin = nn.ReLU
        elif args.halt_nonlin == 'sigmoid':
            nonlin = nn.Sigmoid
        elif args.halt_nonlin == 'tanh':
            nonlin = nn.Tanh

        self.halting_predictor = nn.Sequential()
        for n in range(args.halt_layers):
            self.halting_predictor.add_module(
                'map%d' % n, Linear(embed_dim, embed_dim)
            )
            self.halting_predictor.add_module(
                'nonlin%d' % n, nonlin()
            )

        final_map = Linear(embed_dim, num_exits)
        self.halting_predictor.add_module('map', final_map)
        self.reduce = lambda x, *args: torch.mean(x, dim=0)
        
    def forward(self, encoder_out, hard_decision=False):
        # conditioned on the encoder hidden states picks the best exit
        x = encoder_out['encoder_out']
        if self.detach_before_classifier:
            x = x.detach()
        mask = encoder_out['encoder_padding_mask']
        x = self.reduce(x, mask)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.halting_predictor(x)  # B, Ndec
        x = F.log_softmax(x, dim=-1)
        if hard_decision:
            # Return the argmax:
            return torch.argmax(x, dim=-1)
        return x


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=0):
        super().__init__(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, bias)


