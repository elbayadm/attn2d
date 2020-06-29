"""
label : waitk_transformer and legacy_waitk_transformer
"""
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    gelu, LayerNorm,
    PositionalEmbedding,
)

from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder, 
    FairseqIncrementalDecoder, 
    register_model,
    register_model_architecture,
)

from examples.simultaneous.modules import TransformerEncoderLayer, TransformerDecoderLayer


@register_model('waitk_transformer')
class WaitkTransformerModel(FairseqEncoderDecoderModel):
    """
    Waitk-Transformer with a uni-directional encoder
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder.forward_train(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def decide(self, prev_output_tokens, src_tokens, encoder_out):
        x, _ = self.decoder(prev_output_tokens, encoder_out)
        x = utils.softmax(x[:, -1:], dim=-1)
        x = torch.max(x)
        return  x

    def get_lenx(self, encoder_out):
        return encoder_out['encoder_out'].size(0)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def upgrade_state_dict(self, state_dict):
        if 'encoder.version' in state_dict:
            del state_dict['encoder.version']
        if 'decoder.version' in state_dict:
            del state_dict['decoder.version']

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-inner-dim', type=int, metavar='N',
                            help='decoder qkv projection dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--encoder-decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads for encoder interaction')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--waitk', type=int, 
                            help='wait-k for incremental reading')
        parser.add_argument('--min-waitk', type=int, 
                            help='wait-k for incremental reading')
        parser.add_argument('--max-waitk', type=int, 
                            help='wait-k for incremental reading')
        parser.add_argument('--multi-waitk', action='store_true',  default=False,)


    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def share_embeddings(cls, args, src_dict, tgt_dict):
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError(
                    '--share-all-embeddings requires a joined dictionary'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires \
                    --encoder-embed-dim to match --decoder-embed-dim'
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise RuntimeError('--share-all-embeddings not compatible with \
                                   --decoder-embed-path')
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        return encoder_embed_tokens, decoder_embed_tokens

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder_embed_tokens, decoder_embed_tokens = cls.share_embeddings(args, src_dict, tgt_dict)
        encoder = UnidirTransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return WaitkTransformerModel(encoder, decoder)


class UnidirTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths=None, mask=None, **kwargs):
        """
        Args: src_tokens (batch, src_len)
              src_lengths (batch) 
        Returns:
            dict: - **encoder_out** (src_len, batch, embed_dim)
                  - **encoder_padding_mask**  (batch, src_len)
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        if mask is None:
            mask = torch.triu(utils.fill_with_neg_inf(x.new(x.size(0), x.size(0))), 1)
        for layer in self.layers:
            # Make the encoder unidirectional
            x = layer(
                x, encoder_padding_mask,
                self_attn_mask=mask,
            )

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def slice_encoder_out(self, encoder_out, context_size):
        """ Reorder encoder output according to *new_order*.  """
        sliced_encoder = {}
        if encoder_out['encoder_out'] is not None:
            sliced_encoder['encoder_out'] = encoder_out['encoder_out'].clone()[:context_size]
        else:
            sliced_encoder['encoder_out'] = None
        if encoder_out['encoder_padding_mask'] is not None:
            sliced_encoder['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].clone()[:, :context_size]
        else:
            sliced_encoder['encoder_padding_mask'] = None
        return sliced_encoder

    def reorder_encoder_out(self, encoder_out, new_order):
        """ Reorder encoder output according to *new_order*.  """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


class TransformerDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.waitk = args.waitk
        self.min_waitk = args.min_waitk
        self.max_waitk = args.max_waitk
        self.multi_waitk = args.multi_waitk
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx

        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim) 

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def predict(self, x):
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)
        return x

    def get_attention_mask(self, x, src_len, waitk=None):
        if waitk is None:
            if self.multi_waitk:
                assert self.min_waitk <= self.max_waitk
                waitk = random.randint(min(self.min_waitk, src_len),
                                       min(src_len, self.max_waitk))
            else:
                waitk = self.waitk

        if waitk < src_len:
            encoder_attn_mask = torch.triu(
                utils.fill_with_neg_inf(
                    x.new(x.size(0), src_len)
                ), waitk
            )
            if waitk <= 0:
                encoder_attn_mask[:, 0] = 0

        else:
            encoder_attn_mask = None
        return encoder_attn_mask

    def forward_path(self, prev_output_tokens, encoder_out, waitk=1):
        positions = self.embed_positions(prev_output_tokens) if self.embed_positions is not None else None
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # encoder attn mask following the reading/writing schedule len_tgt x len_src
        encoder_states = encoder_out['encoder_out']  # len_src, B, C
        encoder_attn_mask = self.get_attention_mask(x, encoder_states.size(0), waitk)
        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_states,
                encoder_out['encoder_padding_mask'],
                encoder_attn_mask=encoder_attn_mask,
                self_attn_mask=self.buffered_future_mask(x),
            )

        if self.layer_norm:
            x = self.layer_norm(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.predict(x)
        return x

    def forward_train(self, prev_output_tokens, encoder_out=None, **kwargs):
        positions = self.embed_positions(prev_output_tokens) if self.embed_positions is not None else None
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # encoder attn mask following the reading/writing schedule len_tgt x len_src
        encoder_states = encoder_out['encoder_out']  # len_src, B, C
        encoder_attn_mask = self.get_attention_mask(x, encoder_states.size(0))
        # decoder layers
        for e, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_states,
                encoder_out['encoder_padding_mask'],
                encoder_attn_mask=encoder_attn_mask,
                self_attn_mask=self.buffered_future_mask(x),
            )

        if self.layer_norm:
            x = self.layer_norm(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.predict(x)
        return x, {'attn': attn}

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, cache_decoder=True, **kwargs):
        # Evaluation.
        incremental_state = None
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # decoder layers
        for e, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                encoder_attn_mask=None,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # if incremental_state is not None:
        # Project only the last token
        x = x[-1:]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.predict(x)
        return x, {'attn': attn}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('waitk_transformer', 'waitk_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_inner_dim = getattr(args, 'decoder_inner_dim', args.decoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.encoder_decoder_attention_heads = getattr(args, 'encoder_decoder_attention_heads', args.decoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.gelu = getattr(args, 'gelu', False)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.waitk = getattr(args, 'waitk', 1024) # wait-until-end
    args.min_waitk = getattr(args, 'min_waitk', 1)
    args.max_waitk = getattr(args, 'max_waitk', 1024)


@register_model_architecture('waitk_transformer', 'waitk_transformer_small')
def waitk_transformer_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.encoder_decoder_attention_heads = getattr(args, 'encoder_decoder_attention_heads', args.decoder_attention_heads)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    base_architecture(args)


@register_model_architecture('waitk_transformer', 'waitk_transformer_base')
def waitk_transformer_base(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.encoder_decoder_attention_heads = getattr(args, 'encoder_decoder_attention_heads', args.decoder_attention_heads)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    base_architecture(args)


@register_model_architecture('waitk_transformer', 'waitk_transformer_big')
def waitk_transformer_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.encoder_decoder_attention_heads = getattr(args, 'encoder_decoder_attention_heads', args.decoder_attention_heads)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    base_architecture(args)


