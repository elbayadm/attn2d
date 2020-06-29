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


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class HMMControls3(nn.Module):
    """
    HMM based controller
    """

    def __init__(self, args, controller_dim):

        nn.Module.__init__(self)
        self.gate = nn.Linear(controller_dim, 1, bias=True)
        nn.init.normal_(self.gate.weight, 0, 1/controller_dim)
        nn.init.constant_(self.gate.bias, 0)
        self.detach = args.detach_controls
        
    def get_transitions(self, controls):
        """
        Inputs:
            controls:  log(rho) & log(1-rho)  read/write probabilities: (Tt, B, Ts, 2)
        Returns the log-transition matrix (Tt, B, Ts, Ts)
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
        # print('Controls p(read)', torch.exp(controls[:,:,:,0]).round().data)
        # print('M(t=0)', torch.exp(M[0,0]).data)
        # print('M(t=ly)', torch.exp(M[-1,0]).data)
        # print('Sum transitions:', M.exp().sum(dim=-1))
        return M
    
    def predict_read_write(self, x):
        """ Returns log(rho), log(1-rho) in B, Tt, Ts, 2 """
        if self.detach:
            x = self.gate(x.detach())
        else:
            x = self.gate(x)
        s = F.logsigmoid(x)
        return torch.cat((s, s-x), dim=-1).float()

    def _forward_alpha(self, emissions, M):
        Tt, B, Ts = emissions.size()
        alpha = utils.fill_with_neg_inf(torch.empty_like(emissions))  # Tt, B, Ts
        # initialization  t=1
        # initial = torch.empty_like(alpha[0]).fill_(-math.log(Ts))  # log(1/Ts)
        initial = utils.fill_with_neg_inf(torch.empty_like(alpha[0])) 
        initial[:, 0] = 0
        alpha[0] = emissions[0] + initial
        # print('Initialize alpha:', alpha[0])
        # induction
        for i in range(1, Tt):
            alpha[i] = torch.logsumexp(alpha[i-1].unsqueeze(-1) + M[i-1], dim=1)
            alpha[i] = alpha[i] + emissions[i]
            # print('Emissions@', i, emissions[i])
            # print('alpha@',i, alpha[i])
        return alpha

    def _backward_beta(self, emissions, M):
        Tt, B, Ts = emissions.size()
        beta = utils.fill_with_neg_inf(torch.empty_like(emissions))  # Tt, B, Ts
        # initialization
        beta[-1] = 0
        for i in range(Tt-2, -1, -1):
            beta[i] = torch.logsumexp(M[i].transpose(1, 2) +  # N, Ts, Ts
                                      beta[i+1].unsqueeze(-1) +  # N, Ts, 1
                                      emissions[i+1].unsqueeze(-1),  # N, Ts, 1
                                      dim=1)
        return beta

    def forward(self, observations, emissions):
        """
        Inputs: 
            observations : Input for the controller, B, Tt, Ts, C
            emissions: Output emissions, B, Tt, Ts

        """
        controls = self.predict_read_write(observations).permute(1,0,2,3)  # Tt,B,Ts,2
        Tt, B, Ts = emissions.size()
        # E-step
        with torch.no_grad():
            # get transition matrix:
            M = self.get_transitions(controls.clone())  # Tt, B, Ts, Ts
            # Normalize emissions:
            # print('Emissions:', emissions[:,0].exp().data)
            print('The best context:', torch.argmax(emissions, dim=-1))
            emissions = emissions - emissions.max(dim=-1, keepdim=True)[0]
            print('The best context after norm:', torch.argmax(emissions, dim=-1))

            # print('Div max:', emissions.data)
            # print('Div max(exp)', emissions.exp().data)
            alpha = self._forward_alpha(emissions, M)
            beta = self._backward_beta(emissions, M)
            prior = torch.logsumexp(alpha[-1:], dim=-1, keepdim=True)

            # Sanity check:
            # prior_1 = torch.sum(torch.exp(alpha[1]) * torch.exp(beta[1]), dim=-1)
            # prior_2 = torch.sum(torch.exp(alpha[2]) * torch.exp(beta[2]), dim=-1)
            # print('Prior with n=1:', prior_1, 'Prior with n=2', prior_2, 'Prior with n=-1:', torch.exp(prior.squeeze(-1)))
            
            # Posteriors:
            gamma = alpha + beta - prior
            gamma = torch.exp(gamma)  # Tt, N, Ts
            ksi = alpha[:-1].unsqueeze(-1) + beta[1:].unsqueeze(-2) + emissions[1:].unsqueeze(-2) + M[:-1] - prior.unsqueeze(-1)
            ksi = torch.exp(ksi)

            # Get read/write labels from posteriors:
            write = gamma[1:]
            read = torch.ones_like(write)
            for t in range(1, Tt):
                for j in range(Ts):
                    read[t-1, :, j] = ksi[t-1, :, :j+1, j+1:].sum(dim=-1).sum(dim=-1)
        return controls[:-1], gamma, read, write

