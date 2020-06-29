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


class LLControls(nn.Module):
    """
    LL based controller
    """

    def __init__(self, args, controller_dim):

        nn.Module.__init__(self)
        self.gate = nn.Linear(controller_dim, 1, bias=True)
        nn.init.normal_(self.gate.weight, 0, 1/controller_dim)
        nn.init.constant_(self.gate.bias, 0)
        self.penalty = args.oracle_penalty
        self.write_right = args.write_right
        
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
        with torch.no_grad():
            if self.penalty:
                # Penalize large contexts:
                indices = torch.arange(
                    Ts,
                    dtype=scores.dtype,
                    device=scores.device
                ) / Ts
                scores = scores - self.penalty * indices.unsqueeze(0).unsqueeze(0)
            best_context = scores.max(-1)[1]  # B, Tt
            best_context = progressive_max(best_context).type_as(best_context)
            AP = best_context.float().mean(dim=1) / Ts
            print('AP:', ' '.join(map(lambda x: '{:.2f}'.format(x), AP.tolist())))
            gamma = torch.zeros_like(scores).scatter_(-1, best_context.unsqueeze(-1), 1.0)  # B, Tt, Ts
            if self.write_right:
                gamma = gamma.cumsum(dim=-1)

        # Write beyond the ideal context
        if self.write_right:
            write = gamma[:, 1:]  # B, Tt-1, Ts
        else:
            write = gamma[:, 1:].cumsum(dim=-1)  # B, Tt-1, Ts
        read = 1 - write
        return controls[:, :-1], gamma, read, write
