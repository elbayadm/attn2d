import torch
import torch.nn as nn


class SimulTransOracle(nn.Module):
    """
    Forward-backward based controller
    """

    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty
        
    def forward(self, scores):
        """
        Inputs: Scores : log p(y_t | x<j)  : B, Tt, Ts
        """
        B, Tt, Ts = scores.size()
        # Take negative-likelihood:
        scores = - scores
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
        bs[:, :-1, -1] = torch.cumsum(scores[...,0], dim=-1).flip(-1) 
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
                if cs[b, t+1, j] < cs[b, t, j+1]: # write
                    best.append(j)
                    t += 1
                else:  # read
                    j += 1
            while len(best) < Tt:
                best.append(Ts-1)
            best_context.append(best)

        best_context = torch.stack([torch.Tensor(ctx) for ctx in best_context], dim=0).to(scores.device).long() 
        return best_context
