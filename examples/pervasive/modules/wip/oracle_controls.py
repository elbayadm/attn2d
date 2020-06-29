import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


class LikelihoodOracleControls(nn.Module):
    """
    Oracle based on the likelihood at different contexts
    """
    def __init__(self, input_dim, args):
        super().__init__()
        # 1 (default) : look only at the next context, otherwise look ahead.
        self.look_ahead_margin = args.look_ahead_margin
        self.gate = nn.Linear(input_dim, 1)
        self.detach = args.detach_controls
        self.score_tolerance = args.score_tolerance
        self.infer_gamma_from_read = args.infer_gamma_from_read
        self.normalize_rw = args.normalize_rw
        self.mode = args.oracle_mode
        self.penalty = args.oracle_penalty
        # initialize the gate
        if args.init_control_gate:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0.)

    def fill_controls_emissions_grid(self, controls, emissions, indices, src_length):
        """
        Return controls (C) and emissions (E) covering all the grid
        C : Tt, N, Ts, 2
        E : Tt, N, Ts
        """
        N = controls[0].size(0)
        tgt_length = len(controls)
        gamma = controls[0].new_zeros((tgt_length, src_length, N))
        Cread = controls[0].new_zeros((tgt_length, src_length, N, 1))
        Cwrite = utils.fill_with_neg_inf(torch.empty_like(Cread))
        triu_mask = torch.triu(controls[0].new_ones(tgt_length, src_length), 1).byte()
        triu_mask = triu_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1)
        Cwrite.masked_fill_(triu_mask, 0)
        C = torch.cat((Cread, Cwrite), dim=-1)
        E = utils.fill_with_neg_inf(emissions[0].new(tgt_length, src_length, N))
        for t, (subC, subE) in enumerate(zip(controls, emissions)):
            select = [indices[t].to(C.device)]
            C[t].index_put_(select, subC.transpose(0, 1))
            E[t].index_put_(select, subE.transpose(0, 1))
            gamma[t].index_fill_(0, select[0], 1)
        # Normalize gamma:
        gamma = gamma / gamma.sum(dim=1, keepdim=True)
        return C.transpose(1, 2), E.transpose(1, 2), gamma.transpose(1, 2)

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
        if self.mode == 'read':
            return self.forward_reads(x, emissions, indices, src_length, src_mask)
        elif self.mode == 'gamma':
            return self.forward_gamma(x, emissions, indices, src_length, src_mask)

    def forward_gamma(self, x, emissions, indices, src_length, src_mask):
        controls = [self.logsigmoid_pair(sub) for sub in x] 
        controls, emissions, gamma_unif = self.fill_controls_emissions_grid(controls, emissions, indices, src_length)
        Tt, N, Ts = emissions.size()
        with torch.no_grad():
            # Penalize
            gamma = emissions - self.penalty * torch.arange(Ts).view(1, 1, Ts).type_as(emissions)
            # Exponentiate
            gamma = torch.exp(gamma)
            # normalize
            gamma = gamma / gamma.sum(dim=-1, keepdim=True)
            # print('Gamma:', gamma[:,0])
        write = gamma[1:]
        repartition = torch.cumsum(gamma, dim=-1)[:-1]
        read = repartition - write
        if self.infer_gamma_from_read:
            return emissions, gamma, controls[:-1], read, write
        return emissions, gamma_unif, controls[:-1], read, write

    def forward_reads(self, x, emissions, indices, src_length, src_mask):
        controls = [self.logsigmoid_pair(sub) for sub in x] 
        controls, emissions, gamma_unif = self.fill_controls_emissions_grid(controls, emissions, indices, src_length)
        Tt, N, Ts = emissions.size()
        with torch.no_grad():
            read = torch.zeros_like(emissions)  # 1 for read and 0 for write
            for t in range(Tt):
                # compare contexts:
                for j in range(Ts-1):
                    upto = min(j + self.look_ahead_margin, Ts)
                    if self.score_tolerance:
                        if torch.isinf(emissions[t, 0, j]):  # -inf
                            read[t, :, j] = 1
                        else:
                            read[t, :, j] = (
                                emissions[t, :, j+1:upto+1].max(dim=-1)[0] - emissions[t, :, j]
                            ) / emissions[t, :, j] > self.score_tolerance  # better not write with j
                    else:
                        read[t, :, j] = emissions[t, :, j] < emissions[t, :, j+1:upto+1].max(dim=-1)[0]  # better not write with j

        gamma = 1 - read
        if self.normalize_rw:
            gamma = gamma / gamma.sum(dim=-1, keepdim=True)
        write = gamma[1:]
        read = 1 - write

        if self.infer_gamma_from_read:
            return emissions, gamma, controls[:-1], read, write
        return emissions, gamma_unif, controls[:-1], read, write


