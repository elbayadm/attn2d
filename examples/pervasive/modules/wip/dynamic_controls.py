import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


def logsumexp(a, b):
    m = torch.max(a, b)
    return torch.log(torch.exp(a - m) + torch.exp(b - m)) 


class DynamicControls(nn.Module):
    """
    HMM based controller
    """

    def __init__(self, input_dim, args):
        nn.Module.__init__(self)
        self.gate = nn.Linear(input_dim, 1)
        self.detach = args.detach_controls
        # initialize the gate
        self.normalize_rw = args.normalize_rw
        self.before_after = args.before_after
        self.discretize = args.discretize 
        self.bias_emission = args.bias_emission
        self.pairwise_read = args.pairwise_read
        if args.init_control_gate:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.constant_(self.gate.bias, args.init_control_gate_bias)
        
    def get_transitions(self, controls):
        """
        Inputs:
            controls:  log(rho) & log(1-rho)  read/write probabilities: (Tt, N, Ts, 2)
        Returns the log-transition matrix (N, Tt, Ts, Ts)
            k->j :  p(z_t+1 = j | z_t = k) = (1-rho_tj) prod_l rho_tl
        """
        Tt, N, Ts, _ = controls.size()
        # force rho_tTx = 0
        controls[:, :, -1, 0] = - float('inf')
        controls[:, :, -1, 1] = 0
        M = utils.fill_with_neg_inf(controls.new_empty((Tt, N, Ts, Ts)))
        for k in range(Ts):
            for j in range(k, Ts):
                M[:, :, k, j] = controls[:, :, j, 1] + torch.sum(controls[:, :, k:j, 0], dim=-1)
        return M
    
    def fill_controls_emissions_grid(self, controls, emissions, indices, src_length):
        """
        Return controls (C) and emissions (E) covering all the grid
        C : Tt, N, Ts, 2
        E : Tt, N, Ts
        """
        N = controls[0].size(0)
        tgt_length = len(controls)
        Cread = controls[0].new_zeros((tgt_length, src_length, N, 1))
        Cwrite = utils.fill_with_neg_inf(torch.empty_like(Cread))
        triu_mask = torch.triu(controls[0].new_ones(tgt_length, src_length), 1).byte()
        triu_mask = triu_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1)
        Cwrite.masked_fill_(triu_mask, 0)
        C = torch.cat((Cread, Cwrite), dim=-1)
        E = utils.fill_with_neg_inf(emissions[0].new(tgt_length, src_length, N))
        for t, (subC, subE) in enumerate(zip(controls, emissions)):
            select = [indices[t]]
            C[t].index_put_(select, subC.transpose(0, 1))
            E[t].index_put_(select, subE.transpose(0, 1))
        return C.transpose(1, 2), E.transpose(1, 2)

    def logsigmoid_pair(self, x):
        if self.detach:
            x = self.gate(x.detach())
        else:
            x = self.gate(x)
        s = F.logsigmoid(x)
        return torch.cat((s, s-x), dim=-1)

    def forward(self, x, emissions, indices, src_length, src_mask):
        """
        For N sequences in the batch of max_trg_length Tt and src_length Ts
        Inputs: 
            x : decoder states [(N, #ctx, C)  x Tt]
            emissions: Emissions [(N, #ctx) x Tt]  \log p(y_t|z_t=j, ...) 

        """
        controls = [self.logsigmoid_pair(sub) for sub in x]  # [N, #ctx, 1] xTt
        controls, emissions = self.fill_controls_emissions_grid(controls, emissions, indices, src_length) #Tt, N, Ts
        Tt, N, Ts = emissions.size()
        with torch.no_grad():
            # get transition matrix:
            M = self.get_transitions(controls.clone())  # Tt, N, Ts, Ts
            # Forward
            alpha = utils.fill_with_neg_inf(torch.empty_like(emissions))
            if self.bias_emission:  # penalize large contexts:
                # print('Unbiased:', emissions[:, 0])
                emissions = emissions - self.bias_emission * torch.arange(Ts).view(1, 1, -1).type_as(emissions).to(emissions)
                # print('Biased :', emissions[:, 0])
            # initialization  t=1
            initial = utils.fill_with_neg_inf(torch.empty_like(alpha[0])) 
            initial[:, 0] = 0
            alpha[0] = emissions[0] + initial
            # induction
            for i in range(1, Tt):
                alpha[i] = torch.logsumexp(alpha[i-1].unsqueeze(-1) + M[i-1], dim=1)
                alpha[i] = alpha[i] + emissions[i]

            # Backward
            beta = torch.empty_like(alpha).fill_(-float('inf'))
            # initialization
            beta[-1] = 0
            for i in range(Tt-2, -1, -1):
                beta[i] = torch.logsumexp(M[i].transpose(1, 2) +  # N, Ts, Ts
                                          beta[i+1].unsqueeze(-1) +  # N, Ts, 1
                                          emissions[i+1].unsqueeze(-1),  # N, Ts, 1
                                          dim=1)

            # Sanity check:
            prior = torch.logsumexp(alpha[-1:], dim=-1, keepdim=True)
            # prior_1 = torch.sum(torch.exp(alpha[1]) * torch.exp(beta[1]), dim=-1)
            # prior_2 = torch.sum(torch.exp(alpha[2]) * torch.exp(beta[2]), dim=-1)
            # print('Prior with n=1:', prior_1, 'Prior with n=2', prior_2, 'Prior with n=-1:', torch.exp(prior.squeeze(-1)))
            # print('Alpha:', alpha[:, 0].exp())
            # print('Beta:', beta[:, 0].exp())

            gamma = alpha + beta - prior
            gamma = torch.exp(gamma)  # Tt, N, Ts
            ksi = alpha[:-1].unsqueeze(-1) + beta[1:].unsqueeze(-2) + emissions[1:].unsqueeze(-2) + M[:-1] - prior.unsqueeze(-1)
            ksi = torch.exp(ksi)
            # print('Sum Ksi:', ksi.sum(dim=-1).sum(dim=-1))
            # print('Sum gamma:', gamma.sum(dim=-1))

            # if self.discretize: # binarize r/w labels
                # write = gamma[1:]
                # write = write.ge(self.discretize)
                # read = 1 - write

            if self.before_after: # binarize r/w labels
                gamma = torch.cumsum(gamma, dim=-1)

            write = gamma[1:]
            read = torch.ones_like(write)
            for t in range(1, Tt):
                for j in range(Ts):
                    read[t-1, :, j] = ksi[t-1, :, :j+1, j+1:].sum(dim=-1).sum(dim=-1)
            print('Write summed:', write.sum(dim=-1))
            print('Read summed:', read.sum(dim=-1))

                # if self.normalize_rw:
                    # denom = read + write
                    # mask = denom.eq(0)
                    # read = read / denom
                    # write = write / denom
                    # read[mask] = 0
                    # write[mask] = 0

            # elif self.before_after: #
                # before = torch.cumsum(gamma, dim=-1)  # p(z_t<=j)
                # write = before[1:]
                # read = 1 - before[1:]
            # else: 
                # write = gamma[1:]
                # repartition = torch.cumsum(gamma, dim=-1)[:-1]  # q(z_t <= j) = R_tj + W_tj
                # if self.normalize_rw:
                    # write = write / (repartition + 1e-6)
                    # read = 1 - write
                # else:
                    # read = repartition - write

        return emissions, gamma, controls[:-1], read, write

