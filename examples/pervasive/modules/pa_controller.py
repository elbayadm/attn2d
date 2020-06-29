import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
)
from .tiny_resnet import TinyResNet
from .oracle import SimulTransOracle


class PAController(nn.Module):

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()
        self.args = args
        self.share_embeddings = args.control_share_embeddings
        self.remove_writer_dropout = args.control_remove_writer_dropout

        if not self.share_embeddings: # The controller has its own embeddings
            src_embed_dim = args.control_embed_dim
            tgt_embed_dim = args.control_embed_dim
            self.src_embed_tokens = Embedding(len(src_dict), src_embed_dim, src_dict.pad())
            self.src_embed_scale = math.sqrt(src_embed_dim)
            self.tgt_embed_tokens = Embedding(len(tgt_dict), tgt_embed_dim, tgt_dict.pad())
            self.src_embed_positions = PositionalEmbedding(
                args.max_source_positions,
                src_embed_dim,
                src_dict.pad(),
                learned=args.learned_pos,
            ) if args.control_add_positional_embeddings else None
            self.tgt_embed_positions = PositionalEmbedding(
                args.max_target_positions,
                tgt_embed_dim,
                tgt_dict.pad(),
                learned=args.learned_pos,
            ) if args.control_add_positional_embeddings else None

            self.tgt_embed_scale = math.sqrt(tgt_embed_dim)
            self.embedding_dropout = nn.Dropout(args.control_embeddings_dropout)
            num_features  = src_embed_dim + tgt_embed_dim
        else:
            num_features = args.encoder_embed_dim + args.decoder_embed_dim

        self.net = TinyResNet(num_features,
                              bottleneck=args.bottleneck,
                              ffn_dim=args.ffn_dim,
                              num_layers=args.control_num_layers,
                              kernel_size=args.control_kernel_size,
                              drop_rate=args.control_convolution_dropout,
                              div=args.divide_channels,
                              add_conv_relu=args.add_conv_relu,
                              bias=args.conv_bias,
                              groups=args.conv_groups,
                             )
        
        num_features = self.net.output_channels
        # Oracle:
        self.oracle = SimulTransOracle(
            args.control_oracle_penalty
        ) 

        # Agent : Observation >> Binary R/W decision
        self.gate_dropout = nn.Dropout(args.control_gate_dropout)
        self.gate = nn.Linear(num_features, 1, bias=True)
        nn.init.normal_(self.gate.weight, 0, 1 / num_features)
        nn.init.constant_(self.gate.bias, 0)
        self.write_right = args.control_write_right

        
    @staticmethod
    def add_args(parser):
        parser.add_argument('--control-share-embeddings', action='store_true', default=False)
        parser.add_argument('--control-remove-writer-dropout', action='store_true', default=False)

        parser.add_argument('--control-embed-dim', type=int, default=128)
        parser.add_argument('--control-embeddings-dropout', type=float, default=0.1)

        parser.add_argument('--control-add-positional-embeddings', default=False, action='store_true')

        parser.add_argument('--control-convolution-dropout', type=float, default=0.1)
        parser.add_argument('--control-kernel-size', type=int, help='kernel size', default=3)
        parser.add_argument('--control-num-layers', type=int, help='number of layers', default=8)
        parser.add_argument('--control-gate-dropout', type=float, default=0.)
        parser.add_argument('--control-detach', action='store_true', default=False)
        parser.add_argument('--control-oracle-penalty', type=float, default=0.)
        parser.add_argument('--control-write-right', action='store_true', default=False)

    def observation_grid(self, src_tokens, prev_output_tokens):
        # Build the full grid
        tgt_emb = self.tgt_embed_scale * self.tgt_embed_tokens(prev_output_tokens)
        if self.tgt_embed_positions is not None:
            tgt_emb += self.tgt_embed_positions(prev_output_tokens)
        tgt_emb = self.embedding_dropout(tgt_emb)

        src_emb = self.src_embed_scale * self.src_embed_tokens(src_tokens)
        if self.src_embed_positions is not None:
            src_emb += self.src_embed_positions(src_tokens)
        src_emb = self.embedding_dropout(src_emb)

        batch_size = src_emb.size(0)
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        return x

    def forward(self, sample, encoder_out, decoder_out):
        # First encode the observations
        if not self.share_embeddings:
            x = self.observation_grid(sample['src_tokens'],
                                      sample['prev_output_tokens']) 
        else:
            # The writing input grid
            x = decoder_out[1].clone()

        # Cumulative ResNet:
        x =  self.net(x)
        # Cell aggregation
        # The R/W decisions:
        x = self.gate_dropout(x)
        x = self.gate(x)
        s = F.logsigmoid(x)
        RWlogits = torch.cat((s, s-x), dim=-1).contiguous().float()

        with torch.no_grad():
            lprobs = decoder_out[0].clone()
            target = sample['target']
            encoder_mask = encoder_out['encoder_padding_mask']
            decoder_mask = decoder_out[2]
            # Gather the ground truth likelihoods
            B, Tt, Ts, V = lprobs.size()
            lprobs = utils.log_softmax(lprobs, dim=-1)
            scores = lprobs.view(-1, V).gather(
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)  # BTtTs
            ).view(B, Tt, Ts)
            # Forbid padding positions:  # I'm using NLL beware
            if encoder_mask is not None:
                scores = scores.masked_fill(encoder_mask.unsqueeze(1), -1000)
            if decoder_mask is not None:
                scores = scores.masked_fill(decoder_mask.unsqueeze(-1), -1000)

            # The Oracle
            best_context = self.oracle(scores)

            AP = best_context.add(1).float().mean(dim=1) / Ts
            print('-', round(AP.mean().data.item(), 2))
            Gamma = torch.zeros_like(scores).scatter_(-1, best_context.unsqueeze(-1), 1.0)  # B, Tt, Ts
            
        # Write beyond the ideal context
        if self.write_right:
            Gamma = Gamma.cumsum(dim=-1)
            write = Gamma[:, 1:]  # B, Tt-1, Ts
        else:
            write = Gamma[:, 1:].cumsum(dim=-1)  # B, Tt-1, Ts
        read = 1 - write
        return Gamma, RWlogits[:, :-1], read, write

    def decide(self, src_tokens, prev_output_tokens, writing_grid):
        # torch.set_printoptions(precision=2)
        if not self.share_embeddings:
            x = self.observation_grid(src_tokens,
                                      prev_output_tokens) 
        else:
            x = writing_grid

        # Cumulative ResNet:
        x =  self.net(x)
        # Cell aggreegation
        x = x[:,-1, -1]
        # The R/W decisions:
        x = torch.sigmoid(self.gate(x)).squeeze(-1)  # p(read)
        return  1-x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def _expand(tensor, dim, reps):
    tensor = tensor.unsqueeze(dim)
    shape = tuple(reps if i == dim else -1 for i in range(tensor.dim()))
    return tensor.expand(shape)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


