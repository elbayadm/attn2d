# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math


import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.utils as vutils
from fairseq import utils


from . import (
    FairseqModel, FairseqEncoder, FairseqIncrementalDecoder, 
    register_model, register_model_architecture,
)

from fairseq.modules import (
    DenseNet_CP,
    ResNet,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    BatchHmmLikeMarginals
)


@register_model('attn2d-sim-hmm')
class Attn2dCPModel(FairseqModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                target, target_lengths):
        encoder_out = self.encoder(src_tokens, src_lengths)
        grid_mask = self.build_grid_mask(src_tokens, prev_output_tokens, self.encoder.padding_idx)
        decoder_out = self.decoder(encoder_out, prev_output_tokens, target, grid_mask, target_lengths)
        return decoder_out

    def build_grid_mask(self, src_tokens, tgt_tokens, pad):
        src_mask = src_tokens.eq(pad)  # N, Ts
        tgt_mask = tgt_tokens.eq(pad)  # N, Tt
        grid_mask = (src_mask.unsqueeze(1) + tgt_mask.unsqueeze(-1)).gt(0)  # N, Tt, Ts: either source or target is padded
        return grid_mask
        

    @staticmethod
    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        """ Embeddings """
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--add-positional-embeddings', default=False, action='store_true',
                            help='if set, enables positional embeddings')
        parser.add_argument('--learned-pos', action='store_true',
                            help='use learned positional embeddings')

        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')

        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--ffn-dim', type=int, 
                            help='ffn dimension')
        parser.add_argument('--reduce-dim', type=int, 
                            help='first conv output dimension')
        parser.add_argument('--double-masked', action='store_true',
                            help='Mask the future source as well')

        parser.add_argument('--conv-groups', type=int,
                            help='convolution groups')
        parser.add_argument('--source-dilation', default=1, type=int, 
                            help='2nd dimension dilation')
        parser.add_argument('--target-dilation', default=1, type=int, 
                            help='1st dimension dilation')
        parser.add_argument('--conv-stride', default=1, type=int, 
                            help='2nd dimension stride')
        parser.add_argument('--maintain-resolution', default=1, type=int, 
                            help='pad so that the output dimension matches the input')
        parser.add_argument('--output-dim', type=int, 
                            help='pre-softmax output dimension')

        parser.add_argument('--embeddings-ln', action='store_true',
                            help='add LN after the embeddings')
        parser.add_argument('--network', type=str, metavar='STR',
                            help='Type of cnv net between denseNet or resNet')

        parser.add_argument('--blocks', type=str, metavar='STR',
                            help='specific architecture that overwrites the kernel, growth...')
        parser.add_argument('--kernel-size', type=int, 
                            help='kernel size')
        parser.add_argument('--bn-size', type=int, 
                            help='bn size in the dense layer')
        parser.add_argument('--growth-rate', type=int, 
                            help='growth rate')
        parser.add_argument('--num-layers', type=int, 
                            help='number of layers')

        parser.add_argument('--convolution-dropout', type=float, metavar='D',
                            help='dropout probability in the conv layers')

        parser.add_argument('--input-dropout', type=float, metavar='D',
                            help='dropout probability on the initial 2d input')
        parser.add_argument('--embeddings-dropout', type=float, metavar='D',
                            help='dropout probability on the embeddings')

        parser.add_argument('--prediction-dropout', type=float, metavar='D',
                            help='dropout on the final prediction layer')
        parser.add_argument('--init-weights', type=str, metavar='STR',
                            help='the type of weight initialiation')
        parser.add_argument('--divide-channels', type=int, metavar='INT',
                            help='the factor to reduce the input channels by')
        parser.add_argument('--skip-last-trans', type=bool,
                            help='whether to transition at the last layer')
        parser.add_argument('--memory-efficient', action='store_true',
                            help='use checkpointing')
        parser.add_argument('--trans-norm', type=bool,
                            help='transition batch norm')
        parser.add_argument('--final-norm', type=bool,
                            help='final batch norm')
        parser.add_argument('--layer1-norm', type=bool,
                            help='first layer batch norm')
        parser.add_argument('--layer2-norm', type=bool,
                            help='second layer batch norm')
        parser.add_argument('--initial-shift', type=int, default=3,
                            help='Initial shift')
        parser.add_argument('--read-normalization', type=str, default='max',
                            help='Normalization of the read/write proba from the softmax over the full vocabulary')
        parser.add_argument('--marginals', type=str, default='hmm',
                            help='How q is estimated')
        parser.add_argument('--fixed-policy', type=str, default='path',
                            help='The type of fixed policy with fixed mnaginals')
        parser.add_argument('--fixed-shift', type=int, default=3,
                            help='Fixed policy shift')
        parser.add_argument('--fixed-delta', type=int, default=3,
                            help='Fixed policy stepsize')
        parser.add_argument('--fixed-catchup', type=int, default=1,
                            help='Fixed policy catchup')
        parser.add_argument('--mask-gamma', action='store_true',
                            help='Mask gamma')
        parser.add_argument('--nonzero-gamma', action='store_true',
                            help='Gamma nonzero everywhere')
        parser.add_argument('--eos-bias', type=float, default=5.,
                            help='Bias for the eos token')
        parser.add_argument('--separate-read', action='store_true',
                            help='Reading proba idependent from eos proba')

    def log_tensorboard(self, writer, iter):
        pass

    @classmethod
    def build_model(cls, args, task):
        """ Build a new model instance. """
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = Attn2dEncoder(args, src_dict, encoder_embed_tokens)
        decoder = Attn2dDecoder(args, tgt_dict, decoder_embed_tokens)

        return cls(encoder, decoder)

    def max_decoder_positions(self):
        """ Maximum input length supported by the decoder """
        return self.decoder.max_target_positions 


class Attn2dEncoder(FairseqEncoder):
    def __init__(self, args,  dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            self.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None

        self.ln = lambda x: x
        if args.embeddings_ln:
            self.ln = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        
    def forward(self, src_tokens, src_lengths=None):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = self.ln(x)
        x = self.embedding_dropout(x)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return {
            'encoder_out': x, # B, Ts
            'encoder_padding_mask': encoder_padding_mask  # B, Ts
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(0, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


class Attn2dDecoder(FairseqIncrementalDecoder):
    """ Pervasive Attention Model """

    def __init__(self, args,  dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.read_normalization = args.read_normalization

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.eos = dictionary.eos()
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            left_pad=False,
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None
        self.ln = lambda x: x
        if args.embeddings_ln:
            self.ln = nn.LayerNorm(embed_dim, elementwise_affine=True)

        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        self.input_dropout = nn.Dropout(args.input_dropout)
        self.input_channels = args.encoder_embed_dim + args.decoder_embed_dim

        print('Input channels:', self.input_channels)
        if args.network == 'densenet':
            self.net = DenseNet_CP(self.input_channels, args)
        elif args.network == 'resnet':
            self.net = ResNet(self.input_channels, args)
        else:
            raise ValueError('Unknown architecture %s' % args.network)

        self.output_channels = self.net.output_channels
        print('Output channels:', self.output_channels)

        self.decoder_dim = args.decoder_embed_dim
        self.pool = RowPool()
        self.projection = Linear(self.output_channels, self.decoder_dim,
                                 dropout=args.prediction_dropout)
        self.prediction_dropout = nn.Dropout(args.prediction_dropout)

        # V + 1 for read/write decision
        self.vocab_size = len(dictionary)
        self.prediction = Linear(self.decoder_dim, len(dictionary))  # 1 special token for read and eos
        with torch.no_grad():
            self.prediction.bias[self.eos] = args.eos_bias
        if self.share_input_output_embed:
            self.prediction.weight = self.embed_tokens.weight

        self.separate_read = args.separate_read
        if self.separate_read:
            self.read_proba = Linear(self.decoder_dim, 1)
        self.get_controls = True
        if args.marginals == 'hmm':
            self.marginals = BatchHmmLikeMarginals(args)
        elif args.marginals == 'fixed':
            self.marginals = FixedMarginals(args)
            self.get_controls = False
        self.eps = 1e-6

    def get_normalized_probs(self, net_output, log_probs, sample, get_loss=False):  
        # make get_loss = True in training and false in sampling (self.training couldnt be used)
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output[0].float()
        return F.log_softmax(logits, dim=-1)

    def forward(self, encoder_out, prev_output_tokens, target, grid_mask, target_lengths,
                incremental_state=None, **kwargs):
        # source embeddings
        src_emb = encoder_out['encoder_out']  # N, Ts, ds 
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # N, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # N, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # N, Tt, Ts, C=ds+dt
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = self.net(x, incremental_state)  # N, Tt, Ts, C

        # progressive pooling:
        x = self.pool(x, encoder_out['encoder_padding_mask'])  # B x C x Tt x Ts
        # x, _ = x.max(dim=2)  # N, Tt, C
        x = self.projection(x) if self.projection is not None else x  # N, Tt, C
        x = self.prediction_dropout(x)

        if self.get_controls:
            if self.separate_read:
                xr = self.read_proba(x).float()
                read_controls = F.logsigmoid(xr)  # B x Tt x Ts  log(read)
                write_controls = read_controls - xr
                controls = torch.cat((read_controls, write_controls), dim=3)
                x = self.prediction(x)  # B x Tt x Ts x V
                x = F.log_softmax(x.float(), dim=-1)  # Unstable in half-precision
            else:
                x = self.prediction(x)  # B x Tt x Ts x V
                x = F.log_softmax(x.float(), dim=-1)  # Unstable in half-precision
                # Read controls from p(eos)
                read_controls = x[..., self.eos]  # B x Tt x Ts  log(read/eos)
                if self.read_normalization == 'max':
                    denom = torch.max(x, dim=-1)[0]
                    read_controls = read_controls - denom   # log[ p(eos) / max_y p(y) ]
                    write_controls = torch.log(1 - torch.exp(read_controls) + self.eps)  # log(write)
                    controls = torch.stack((read_controls, write_controls), dim=3)

                elif self.read_normalization == 'max_detach':
                    denom = torch.max(x, dim=-1)[0].detach()
                    read_controls = read_controls - denom   # log[ p(eos) / max_y p(y) ]
                    write_controls = torch.log(1 - torch.exp(read_controls) + self.eps)  # log(write)
                    controls = torch.stack((read_controls, write_controls), dim=3)

                elif self.read_normalization == 'count':
                    order = (read_controls.unsqueeze(-1) - x).ge(0).sum(dim=-1)  #read
                    order = order.float() / self.vocab_size
                    read_controls = torch.log(order) + read_controls - read_controls.detach()  # the value of order with the gradient of p(eos)
                    write_controls = torch.log(1 - order + self.eps)  # log(write)
                    controls = torch.stack((read_controls, write_controls), dim=3)
                else:
                    assert self.read_normalization == 'asis'
                    # write_controls = torch.log(1 - torch.exp(read_controls))  # log(write)
                    xother = torch.cat([x[..., :self.eos], x[..., self.eos+1:]], dim=-1)
                    write_controls = torch.logsumexp(xother, dim=-1)
                    controls = torch.stack((read_controls, write_controls), dim=3)
            print('Read:', read_controls.exp()[0].squeeze())
        else:
            controls = None
            x = self.prediction(x)  # B x Tt x Ts x V
            x = F.log_softmax(x.float(), dim=-1)  # Unstable in half-precision

        duplicate_target = target.unsqueeze(-1).expand(-1, -1, x.size(2)).unsqueeze(-1)
        lprobs = x.gather(dim=-1, index=duplicate_target).squeeze(-1)
        with torch.no_grad():
            marginals = self.marginals(emissions=lprobs,
                                       controls=None if controls is None else controls.clone(),
                                       target_lengths=target_lengths)
        return lprobs, controls, marginals

    def forward_one(self, prev_output_tokens, encoder_out, context_size,
                    incremental_state=None, **kwargs):
        # Truncate the encoder outputs:
        encoder_out_truncated = {'encoder_out': encoder_out['encoder_out'].clone()[:,:context_size]}
        # source embeddings
        src_emb = encoder_out_truncated['encoder_out']  # N, Ts, ds 
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # N, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # N, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # N, Tt, Ts, C=ds+dt
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = self.net(x, incremental_state)  # N, Tt, Ts, C

        # progressive pooling:
        # x = self.pool(x, encoder_out['encoder_padding_mask'])  # B x C x Tt x Ts
        x, _ = x.max(dim=2)  # N, Tt, C
        x = self.projection(x) if self.projection is not None else x  # N, Tt, C
        x = self.prediction_dropout(x)
        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # N, Tt, V
        return x, {'attn': None}

    def _forward_one(self, prev_output_tokens, encoder_out,
                    incremental_state=None, **kwargs):
        # prev_output tokens contains the read/eos symbol as well
        B, T = prev_output_tokens.size()
        # pick the last true tgt token:
        # flipped = torch.flip(1 - torch.eq(prev_output_tokens, self.eos), [1])
        # true_tokens = T - 1 - torch.argmax(flipped, dim=-1)
        # # print('True tokens:', true_tokens)
        # prev_output_tokens = prev_output_tokens.gather(1, true_tokens.unsqueeze(-1))
        # print('Gathered:', prev_output_tokens)
        # remove eos tokens:
        # eos_tokens = prev_output_tokens.eq(self.eos)
        # eos_tokens[:, 0] = 0
        # true_tokens = []
        # for e in range(B):
            # true_tokens.append(prev_output_tokens[e].masked_select(1-eos_tokens[e]))
        # prev_output_tokens = torch.stack(true_tokens, dim=0)
        # print('Prev tokens:', prev_output_tokens[:10])


        # source embeddings
        src_emb = encoder_out['encoder_out']  # B x Ts x ds
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B x Tt x Ts x ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B x Tt x Ts x dt
        x = torch.cat((src_emb, tgt_emb), dim=3)  # B x Tt x Ts x C
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = x.permute(0, 3, 1, 2)  # B x C x Tt x Ts
        x = self.net(x, incremental_state)  # B x C x Tt x Ts
        x = x[:, :, -1]  # B, C, Ts

        # Pool over true source tokens
        if encoder_out['encoder_padding_mask'] is not None:
            mask_inf = torch.zeros_like(x)  # B, C, Ts
            # -inf in padded pos
            mask_inf = mask_inf.masked_fill(encoder_out['encoder_padding_mask'].unsqueeze(1), float('-inf'))
            x += mask_inf
        x, _ = x.max(dim=-1)  # B x C
        x = x.unsqueeze(1)  # B x Tt x C
        x = self.projection(x)  # B x Tt x dt
        x = self.prediction_dropout(x)

        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B x Tt x V
        return x, {'attn': None}

    def forward_read_merge(self, prev_output_tokens, encoder_out,
                           incremental_state=None, th=0.5, **kwargs):
        # prev_output tokens contains the read/eos symbol as well
        B, T = prev_output_tokens.size()
        # source embeddings
        src_emb = encoder_out['encoder_out']  # B x Ts x ds
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B x Tt x Ts x ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B x Tt x Ts x dt
        x = torch.cat((src_emb, tgt_emb), dim=3)  # B x Tt x Ts x C
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = x.permute(0, 3, 1, 2)  # B x C x Tt x Ts
        x = self.net(x, incremental_state)  # B x C x Tt x Ts
        x = x[:, :, -1]  # B, C, Ts

        # Pool over true source tokens
        if encoder_out['encoder_padding_mask'] is not None:
            mask_inf = torch.zeros_like(x)  # B, C, Ts
            # -inf in padded pos
            mask_inf = mask_inf.masked_fill(encoder_out['encoder_padding_mask'].unsqueeze(1), float('-inf'))
            x += mask_inf
        x, _ = x.max(dim=-1)  # B x C
        x = x.unsqueeze(1)  # B x Tt x C
        x = self.projection(x)  # B x Tt x dt
        x = self.prediction_dropout(x)
        # read proba:
        read = self.read_proba(x).sigmoid() / 2
        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B x Tt x V
        # merge the read proba as a token proba:
        x = F.softmax(x, dim=-1)
        # print('Read:', read[:5])
        # print('Vocab distrib:', x[:5].max(dim=-1))
        x = (1 - read) * x
        x = torch.cat((x, read), dim=-1)
        x = x[:, -1]
        # print('x:', x[:1])
        return x, {'attn': None}


    def forward_read(self, prev_output_tokens, encoder_out,
                     incremental_state=None, th=0.5, **kwargs):
        # prev_output tokens contains the read/eos symbol as well
        B, T = prev_output_tokens.size()
        # source embeddings
        src_emb = encoder_out['encoder_out']  # B x Ts x ds
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B x Tt x Ts x ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B x Tt x Ts x dt
        x = torch.cat((src_emb, tgt_emb), dim=3)  # B x Tt x Ts x C
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = x.permute(0, 3, 1, 2)  # B x C x Tt x Ts
        x = self.net(x, incremental_state)  # B x C x Tt x Ts
        x = x[:, :, -1]  # B, C, Ts

        # Pool over true source tokens
        if encoder_out['encoder_padding_mask'] is not None:
            mask_inf = torch.zeros_like(x)  # B, C, Ts
            # -inf in padded pos
            mask_inf = mask_inf.masked_fill(encoder_out['encoder_padding_mask'].unsqueeze(1), float('-inf'))
            x += mask_inf
        x, _ = x.max(dim=-1)  # B x C
        x = x.unsqueeze(1)  # B x Tt x C
        x = self.projection(x)  # B x Tt x dt
        x = self.prediction_dropout(x)
        # read proba:
        read = self.read_proba(x).squeeze(-1).squeeze(-1).sigmoid()
        # print('Read:', read.flatten())
        read = read.gt(th)
        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B x Tt x V
        return x, read, {'attn': None}


@register_model_architecture('attn2d-sim-hmm', 'attn2d-sim-hmm')
def base_architecture(args):
    args.memory_efficient = getattr(args, 'memory_efficient', False)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.share_decoder_input_output_embed = getattr(args,
                                                    'share_decoder_input_output_embed',
                                                    False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.embeddings_dropout = getattr(args, 'embeddings_dropout', 0.)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.ffn_dim = getattr(args, 'ffn_dim', 512)
    args.output_dim = getattr(args, 'output_dim', args.decoder_embed_dim)
    args.reduce_dim = getattr(args, 'reduce_dim', args.encoder_embed_dim // 2)

    args.conv_groups = getattr(args, 'conv_groups', 1)
    args.conv_stride = getattr(args, 'conv_stride', 1)
    args.source_dilation = getattr(args, 'source_dilation', 1)
    args.target_dilation = getattr(args, 'target_dilation', 1)
    args.maintain_resolution = getattr(args, 'maintain_resolution', 1)
    
    args.add_positional_emnbeddings = getattr(args, 'add_positional_embeddings', False)
    args.learned_pos = getattr(args, 'learned_pos', False)
    args.embeddings_ln = getattr(args, 'embeddings_ln', False)

    args.input_dropout = getattr(args, 'input_dropout', 0.2)
    args.convolution_dropout = getattr(args, 'convolution_dropout', 0.2)

    args.network = getattr(args, 'network', 'densenet')
    args.read_normalization = getattr(args, 'read_normalization', 'max')


    args.growth_rate = getattr(args, 'growth_rate', 32)
    args.kernel_size = getattr(args, 'kernel_size', 3)
    args.bn_size = getattr(args, 'bn_size', 4)
    args.num_layers = getattr(args, 'num_layers', 24)

    blocks = "[[[%d, %d, %d, %.2f]] * %d]" % (args.growth_rate,
                                              args.bn_size,
                                              args.kernel_size,
                                              args.convolution_dropout,
                                              args.num_layers)

    args.blocks = getattr(args, 'blocks', blocks)
    args.divide_channels = getattr(args, 'divide_channels', 2)
    args.prediction_dropout = getattr(args, 'prediction_dropout', 0.2)
    args.skip_last_trans = getattr(args, 'skip_last_trans', False)
    args.trans_norm = getattr(args, 'trans_norm', False)
    args.final_norm = getattr(args, 'final_norm', False)
    args.layer1_norm = getattr(args, 'layer1_norm', False)
    args.layer2_norm = getattr(args, 'layer2_norm', False)
    args.double_masked = getattr(args, 'double_masked', True)
    args.mask_gamma = getattr(args, 'mask_gamma', False)
    args.nonzero_gamma = getattr(args, 'nonzero_gamma', False)
    args.separate_read = getattr(args, 'separate_read', False)
    args.eos_bias = getattr(args, 'eos_bias', 5.)

def _expand(tensor, dim, reps):
    tensor = tensor.unsqueeze(dim)
    shape = tuple(reps if i == dim else -1 for i in range(tensor.dim()))
    return tensor.expand(shape)


def PositionalEmbedding(num_embeddings, embedding_dim,
                        padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1,
                                       embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad,
                                          num_embeddings + padding_idx + 1)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0., bias=True):
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(m.weight, mean=0,
                    std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return m


class RowPool(nn.Module):
    """
    Progressively pool the feature maps along the rows
    At every position ((i, j) we'll assign MaxPool(X[i, :j])
    """
    def __init__(self, *args):
        super(RowPool, self).__init__()

    def forward(self, X, src_mask):
        N, C, Tt, Ts = X.size()
        Xpool = torch.empty_like(X)
        for i in range(Tt):
            if src_mask is not None:
                mask_inf = torch.zeros_like(X[:, :, i])  # B, C, Ts
                # -inf in padded pos
                mask_inf = mask_inf.masked_fill(src_mask.unsqueeze(1), float('-inf'))
                X[:, :, i] = X[:, :, i] + mask_inf
            for j in range(Ts):
                Xpool[:, :, i, j] = torch.max(X[:, :, i, :j+1], dim=-1)[0]
        return Xpool


class FixedMarginals(nn.Module):
    def __init__(self, args):
        super(FixedMarginals, self).__init__()
        self.shift = args.fixed_shift
        self.delta = args.fixed_delta
        self.catchup = args.fixed_catchup
        self.policy = args.fixed_policy
        self.eps = 1e-6

    def forward(self, emissions, **kwargs):
        if self.policy == 'path':
            return self.forward_single_path(emissions)
        if self.policy == 'above_path':
            return self.forward_above_path(emissions)
        if self.policy == 'diagonal':
            return self.forward_around_diagonal(emissions)
        raise ValueError('Unknown policy ', self.policy)

    def forward_single_path(self, emissions):
        """
        Returns q(z_t) = 1 on the fixed path given by (shift, delta, catchup)
        The first argument is the emissions (N, Tt, Ts, V)
        """
        N, Tt, Ts = emissions.size()
        # Read:
        grid = emissions.new_empty((Tt, Ts)).fill_(self.eps)
        for t in range(Tt):
            ctx = min(self.shift + t // self.catchup * self.delta, Ts - 1)
            grid[t, ctx] = 1
        grid = grid.unsqueeze(0).expand(N, -1, -1)  # N, Tt, Ts
        return {'gamma': grid}

    def forward_above_path(self, emissions):
        """
        Returns q(z_t) = 1 on the fixed path given by (shift, delta, catchup)
        The first argument is the emissions (N, Tt, Ts, V)
        """
        N, Tt, Ts = emissions.size()
        # Read:
        grid = emissions.new_empty((Tt, Ts)).fill_(self.eps)
        for t in range(Tt):
            ctx = min(self.shift + t // self.catchup * self.delta, Ts - 1)
            grid[t, ctx:] = 1 / (Ts - ctx + 1)
        grid = grid.unsqueeze(0).expand(N, -1, -1)  # N, Tt, Ts
        return {'gamma': grid}

    def forward_around_diagonal(self, emissions):
        """
        Returns q(z_t) = 1 on the fixed path given by (shift, delta, catchup)
        The first argument is the emissions (N, Tt, Ts, V)
        """
        N, Tt, Ts = emissions.size()
        # Read:
        grid = emissions.new_empty((Tt, Ts)).fill_(self.eps)
        for t in range(Tt):
            minctx = min(max(t - self.delta, 0), Ts-1)
            maxctx = min(t + self.delta, Ts - 1)
            grid[t, minctx:maxctx+1] = 1 / (maxctx - minctx + 1)
        grid = grid.unsqueeze(0).expand(N, -1, -1)  # N, Tt, Ts
        return {'gamma': grid}

