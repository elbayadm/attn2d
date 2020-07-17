import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils

from . import register_aggregator

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_aggregator('avg')
class AVG(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()

    def forward(self, x, need_attention_weights=False):
        # Avgpool 
        x = x.mean(dim=2)  # B, Tt, C
        return x, None

    def one_step(self, x):
        return self.forward(x)


@register_aggregator('max')
class MAX(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()

    def forward(self, x, need_attention_weights=False):
        if not need_attention_weights:
            # Maxpool 
            x, _ = x.max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            # x in B, Tt, Ts, C
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn

    def one_step(self, x):
        return self.forward(x)


@register_aggregator('gated-max')
class GatedMAX2(nn.Module):
    """ Max on top of GLU (gated-max2)"""
    def __init__(self, args, num_features):
        super().__init__()
        self.linear = Linear(num_features, 2*num_features)

    def forward(self, x, need_attention_weights=False):
        x = F.glu(self.linear(x), dim=-1) # B, Tt, Ts, C
        if not need_attention_weights:
            # Maxpool 
            x, _ = x.max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            # x in B, Tt, Ts, C
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn

    def one_step(self, x):
        return self.forward(x)


@register_aggregator('attn')
class ATTN(nn.Module):
    """
    Attention
    """
    def __init__(self, args, num_features):
        super().__init__()
        self.w1 = Linear(num_features, num_features)
        self.w2 = Linear(num_features, 1)

    def forward(self, x, need_attention_weights=False):
        # Attention scorees:
        alpha = self.w2(self.w1(x))  # B, Tt, Ts, 1
        alpha = utils.softmax(alpha, dim=2).type_as(alpha)
        x = x.permute(0,1,3,2)
        x = torch.matmul(x, alpha).squeeze(-1)
        if need_attention_weights:
            return x, alpha.squeeze(-1)
        return x, None

    def one_step(self, x):
        return self.forward(x)


@register_aggregator('cell')
class UnidirCell(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()

    def one_step(self, x, need_attention_weights=False):
        if not need_attention_weights:
            x = x[:,-1:, -1]
            return x, None
        # Output attention weights:
        return None, None

    def forward(self, x, need_attention_weights=False):
        return x[:, :, -1], None

    def one_step(self, x):
        return self.forward(x)


@register_aggregator('path-max')
class PathMAX(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.waitk = args.waitk

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk", type=int, default=3)

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:] # B,1,Ts,C
        if not need_attention_weights:
            x, _ = x.max(dim=2)  # B, 1, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            return None, None

    def forward(self, x, need_attention_weights=False):
        if not need_attention_weights:
            # Maxpool 
            B, Tt, Ts, C = x.size()
            mask = torch.triu(utils.fill_with_neg_inf(x.new(Tt, Ts)), self.waitk)
            # print('Mask (%d, %d):' % (Tt, Ts), mask)
            # for t in range(Tt):
                 # ctx = min((t // 1 * 1)  + self.waitk, Ts)
                 # print('z_%d = %d' % (t, ctx))
            x, _ = (
                x + mask.unsqueeze(0).unsqueeze(-1)
            ).max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            # x in B, Tt, Ts, C
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn


@register_aggregator('path-gated-max')
class PathGatedMAX(nn.Module):
    """ Max on top of GLU """
    def __init__(self, args, num_features):
        super().__init__()
        self.waitk = args.waitk
        self.linear = Linear(num_features, 2*num_features)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk", type=int, default=3)

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:] # B,1,Ts,C
        x = F.glu(self.linear(x), dim=-1) # B, 1, Ts, C
        if not need_attention_weights:
            x, _ = x.max(dim=2)  # B, 1, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            return None, None

    def forward(self, x, need_attention_weights=False):
        x = F.glu(self.linear(x), dim=-1) # B, Tt, Ts, C
        if not need_attention_weights:
            # Maxpool 
            B, Tt, Ts, C = x.size()
            mask = torch.triu(utils.fill_with_neg_inf(x.new(Tt, Ts)), self.waitk)
            x, _ = (
                x + mask.unsqueeze(0).unsqueeze(-1)
            ).max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            # x in B, Tt, Ts, C
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn


@register_aggregator('path-attn')
class PathATTN(nn.Module):
    """
    Attention
    """
    def __init__(self, args, num_features):
        super().__init__()
        self.waitk = args.waitk
        self.w1 = Linear(num_features, num_features)
        self.w2 = Linear(num_features, 1)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk", type=int, default=3)

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:]   # B, 1, Ts, C
        alpha = self.w2(self.w1(x))  # B, 1, Ts, 1
        alpha = utils.softmax(alpha, dim=2)
        x = x.permute(0,1,3,2)
        x = torch.matmul(x, alpha).squeeze(-1)
        if need_attention_weights:
            return x, alpha.squeeze(-1)
        return x, None

        return self.forward(x, need_attention_weights)
        
    def forward(self, x, need_attention_weights=False):
        # Attention scorees:
        B, Tt, Ts, C = x.size()
        alpha = self.w2(self.w1(x))  # B, Tt, Ts, 1
        mask = torch.triu(utils.fill_with_neg_inf(x.new(Tt, Ts)), self.waitk)
        alpha = utils.softmax(alpha + mask.unsqueeze(0).unsqueeze(-1), dim=2).type_as(alpha)
        x = x.permute(0,1,3,2)
        x = torch.matmul(x, alpha).squeeze(-1)
        if need_attention_weights:
            return x, alpha.squeeze(-1)
        return x, None


@register_aggregator('path-cell')
class PathCell(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.waitk = args.waitk

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk", type=int, default=3)

    def one_step(self, x, need_attention_weights=False):
        if not need_attention_weights:
            x = x[:,-1:, -1]
            return x, None
        # Output attention weights:
        return None, None

    def forward(self, x, need_attention_weights=False):
        if not need_attention_weights:
            # Select the cell corresponding to waitk+t
            B, Tt, Ts, C = x.size()
            indices = (
                torch.arange(Tt, device=x.device) + self.waitk - 1
            ).clamp(max=Ts-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  #1,Tt,1,1
            indices = indices.repeat(B,1,1,C)
            x = x.gather(2, indices)

            return x, None
        # Output attention weights:
        if need_attention_weights:
            return None, None


@register_aggregator('path-max-full')
class PathMAXFull(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.waitk = args.waitk

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk", type=int, default=3)

    def one_step(self, x, need_attention_weights=False):
        if not need_attention_weights:
            x, _ = x.max(dim=2)  # B, Tt, C
            x, _ = x.max(dim=1, keepdim=True)  # B, C
            return x, None
        # Output attention weights:
        return None, None


    def forward(self, x, need_attention_weights=False):
        if not need_attention_weights:
            # Maxpool 
            B, Tt, Ts, C = x.size()
            if Ts > self.waitk:
                x = x.permute(0, 3, 1, 2)  # B, C, Tt, Ts
                x = F.pad(x, (Ts-self.waitk, self.waitk-1, Tt-1, 0), 'constant', -1000)
                x = F.max_pool2d(x, 
                                 (Tt, Ts), # kernel size
                                 (1, 1), # stride
                                 0, # padding
                                 1, # dilation
                                 False, # ceil_mode
                                 False, # return indices
                                )
                indices = torch.arange(Tt, device=x.device).clamp(max=Ts-1)
                indices = indices.view(1,1,Tt,1).repeat(B,C,1,1)
                x = x.gather(-1, indices).squeeze(-1)
                x = x.permute(0, 2, 1)
            else:
                x = x.permute(0, 3, 1, 2)  # B, C, Tt, Ts
                x = F.pad(x, (0, 0, Tt-1, 0), 'constant', -1000)
                x = F.max_pool2d(x, 
                                 (Tt, Ts), # kernel size
                                 (1, 1), # stride
                                 0, # padding
                                 1, # dilation
                                 False, # ceil_mode
                                 False, # return indices
                                )
                x = x[...,0]
                x = x.permute(0, 2, 1)
            return x, None

        # Output attention weights:
        if need_attention_weights:
            return None, None



@register_aggregator('grid-attn')
class GridATTN(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.w1 = Linear(num_features, num_features)
        self.w2 = Linear(num_features, 1)

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:]   # B, 1, Ts, C
        alpha = self.w2(self.w1(x))  # B, 1, Ts, 1
        alpha = utils.softmax(alpha, dim=2)
        x = x.permute(0,1,3,2)
        x = torch.matmul(x, alpha).squeeze(-1)
        if need_attention_weights:
            return x, alpha.squeeze(-1)
        return x, None

    def forward(self, x, need_attention_weights=False):
        # Attention scorees:
        B, Tt, Ts, C = x.size()
        alpha = self.w2(self.w1(x))  # B, Tt, Ts, 1
        # for every (t,j) allow first j
        mask = torch.triu(utils.fill_with_neg_inf(x.new(Ts, Ts)), 1).type_as(alpha)
        alpha = alpha.permute(0,1,3,2) + mask.unsqueeze(0).unsqueeze(0)  # B,Tt,Ts,Ts
        alpha = utils.softmax(alpha, dim=-1)
        x = torch.matmul(alpha, x)
        return x, None


@register_aggregator('grid-max-slow')
class GridMAX2(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()

    def forward(self, x, need_attention_weights=False):
        B, Tt, Ts, C = x.size()
        xpool = torch.zeros_like(x)
        for t in range(Tt):
            for j in range(Ts):
                xpool[:,t,j,:] = torch.max(x[:,t,:j+1,:], dim=-2)[0]
        return xpool, None
        

@register_aggregator('grid-max')
class GridMAX(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:]
        if not need_attention_weights:
            x, _ = x.max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn

    def forward(self, x, need_attention_weights=False):
        B, Tt, Ts, C = x.size()
        x = x.permute(0, 3, 1, 2)  # B, C, Tt, Ts
        x = F.pad(x, (Ts-1, 0), 'constant', -1000)
        x = F.max_pool2d(x, 
                         (1, Ts), # kernel size
                         (1, 1), # stride
                         0, # padding
                         1, # dilation
                         False, # ceil_mode
                         False, # return indices
                        )
        x = x.permute(0, 2, 3, 1)
        return x, None


@register_aggregator('grid-gated-max')
class GridGatedMAX(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.linear = Linear(num_features, 2*num_features)

    def one_step(self, x, need_attention_weights=False):
        x = x[:, -1:]  # B, 1, Ts, C
        x = F.glu(self.linear(x), dim=-1) # B, 1, Ts, C
        if not need_attention_weights:
            x, _ = x.max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn

    def forward(self, x, need_attention_weights=False):
        B, Tt, Ts, C = x.size()
        x = F.glu(self.linear(x), dim=-1) # B, Tt, Ts, C
        x = x.permute(0, 3, 1, 2)  # B, C, Tt, Ts
        x = F.pad(x, (Ts-1, 0), 'constant', -1000)
        x = F.max_pool2d(x, 
                         (1, Ts), # kernel size
                         (1, 1), # stride
                         0, # padding
                         1, # dilation
                         False, # ceil_mode
                         False, # return indices
                        )
        x = x.permute(0, 2, 3, 1)
        return x, None


@register_aggregator('grid-gated-max2')
class GridGatedMAX2(nn.Module):
    """ Max on top of GLU """
    def __init__(self, args, num_features):
        super().__init__()
        self.linear = Linear(num_features, 2*num_features)

    def forward(self, x, need_attention_weights=False):
        x = F.glu(self.linear(x), dim=-1) # B, Tt, Ts, C
        if not need_attention_weights:
            # Maxpool 
            x, _ = x.max(dim=2)  # B, Tt, C
            return x, None
        # Output attention weights:
        if need_attention_weights:
            # x in B, Tt, Ts, C
            B, Tt, Ts, C = x.size()
            x, indices = x.max(dim=2)
            # indices in B, Tt, C with each channel selecting a source position
            # Terrible but will do:
            attn = x.new_zeros(B, Tt, Ts)
            for i in range(Ts):
                attn[:,:,i] = indices.eq(i).sum(dim=-1)
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
        return x, attn



