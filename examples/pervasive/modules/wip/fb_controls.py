import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


def progressive_max(x):
    T = x.size(1)
    x = F.pad(x, (T-1, 0), 'constant', -1)
    x = F.max_pool1d(x.unsqueeze(1).float(),  # shape into B, C, T
                    T, # kernel size
                    1, # stride
                    0, # padding
                    1, # dilation
                    False, # ceil_mode
                    False, # return indices
                    )
    return x.squeeze(1)  # B, Tt

def logsumexp(a, b):
    m = torch.max(a, b)
    return torch.log(torch.exp(a - m) + torch.exp(b - m)) 


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class FBControls(nn.Module):
    """
    Forward-backward based controller
    """

    def __init__(self, args, controller_dim):

        nn.Module.__init__(self)
        self.gate = nn.Linear(controller_dim, 1, bias=True)
        nn.init.normal_(self.gate.weight, 0, 1/controller_dim)
        nn.init.constant_(self.gate.bias, 0)
        self.penalty = args.oracle_penalty
        self.write_right = args.write_right
        self.delta_control = args.delta_control

        
    def get_positions_proba(self, rw_logits):
        """
        Inputs:
            rw_logits [log(rho), log(1-rho)]  : (Tt, B, Ts, 2)
        Returns the probabilities of being at position (t,j) (Tt, B, Ts)
        """
        Tt, B, Ts, _ = rw_logits.size()
        Mr1 = rw_logits[0:1,:,:-1,0].exp()
        Mc1 = rw_logits[:,:,0:1,1].exp()
        M = rw_logits[1:,:,:-1,0].exp() + rw_logits[:-1,:,1:,1].exp()
        M = torch.cat((Mr1, M), dim=0)
        M = torch.cat((Mc1, M), dim=-1)
        return M
    
    def predict_read_write(self, x):
        """ Returns log(rho), log(1-rho) in B, Tt, Ts, 2 """
        x = self.gate(x)
        s = F.logsigmoid(x)
        return torch.cat((s, s-x), dim=-1).float()

    def forward(self, observations, scores):
        """
        Inputs: 
            observations : Input for the controller: B, Tt, Ts, C
            Scores : log p(y_t | x<j)  : B, Tt, Ts
        """
        controls = self.predict_read_write(observations)  # B,Tt,Ts,2
        B, Tt, Ts = scores.size()
        # Take negative-likelihood:
        scores = - scores
        with torch.no_grad():
            # Estime the best decoding path i.e context sizes
            # Forwad pass costs
            fs = scores.new_zeros(B, Tt+1, Ts)
            # First column:
            fs[:, 1:, 0] = torch.cumsum(scores[..., 0], dim=-1)
            # First row:
            fs[:, 0] =  self.penalty * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(fs).unsqueeze(0).repeat(B, 1) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = fs[:, t-1, j] + scores[:, t-1, j]  # Write (t-1, j) -> (t, j)
                    ifread = fs[:, t, j-1] + self.penalty * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    fs[:, t, j] = torch.min(
                        ifwrite, ifread
                    )
            bs = scores.new_zeros(B, Tt+1, Ts)
            # Last column:
            bs[:, :-1, -1] = torch.cumsum(scores[...,-1], dim=-1).flip(-1) 
            # Last row:
            bs[:, -1] =  self.penalty * torch.arange(1,  Ts+1).type_as(bs).unsqueeze(0).repeat(B, 1).flip(-1) / Ts
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = bs[:, t+1, j] + scores[:, t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = bs[:, t, j+1] + self.penalty * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    bs[:, t,j] = torch.min(
                        ifwrite, ifread
                    )
            # accumulate the scores
            cs = fs + bs
            best_context = []
            for b in range(B):
                t = 0 
                j = 0
                best = []
                while t < Tt and j < Ts-1:
                    if cs[b, t+1, j] <= cs[b, t, j+1]: # write
                        best.append(j)
                        t += 1
                    else:  # read
                        j += 1
                while len(best) < Tt:
                    best.append(Ts-1)
                best_context.append(best)
            best_context = torch.stack(
                [torch.Tensor(ctx) for ctx in best_context], dim=0
            ).to(scores.device).long()
            AP = best_context.add(1).float().mean(dim=1) / Ts
            print('AP:', ' '.join(map(lambda x: '{:.2f}'.format(x), AP.tolist())))
            gamma = torch.zeros_like(scores).scatter_(-1, best_context.unsqueeze(-1), 1.0)  # B, Tt, Ts

        if self.write_right and not self.delta_control:  # maximal
            # Default: seems to work fine
            # Optimize the writer above the ideal path
            # W/R right and left of the oracle path:
            gamma = gamma.cumsum(dim=-1)
            write = gamma[:, 1:]  # B, Tt-1, Ts
            read = 1 - write

        if self.write_right and self.delta_control:  
            # Optimize the writer right of the ideal path
            # W/R along the oracle path:
            write = gamma[:, 1:]
            gamma = gamma.cumsum(dim=-1)

        if not self.write_right  and self.delta_control:  #minimal
            # Optimize the writer on the oracle path
            # W/R along the oracle path:
            write = gamma[:, 1:].cumsum(dim=-1)  # B, Tt-1, Ts
            read = 1 - write

        if not self.write_right  and not self.delta_control:
            # Optimize the writer on the oracle path
            # W/R right and left of the oracle path:
                read = gamma


        return controls[:, :-1], gamma, read, write
