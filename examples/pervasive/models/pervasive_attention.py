import torch
import torch.nn as nn
from fairseq import utils

from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder, FairseqIncrementalDecoder, 
    register_model, register_model_architecture,
)

from examples.pervasive.modules import (
    build_convnet, build_aggregator,
)

from fairseq.modules import (
    PositionalEmbedding,
)


@register_model('pervasive')
class PervasiveModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        """ Embeddings """
        parser.add_argument('--skip-output-mapping', action='store_true',
                            help='remove the final mapping if equal dimension')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share-decoder-input-output-embed', 
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--add-positional-embeddings', 
                            default=False, 
                            action='store_true',
                            help='if set, enables positional embeddings')
        parser.add_argument('--learned-pos', 
                            action='store_true', default=False,
                            help='use learned positional embeddings')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--output-dim', type=int, 
                            help='pre-softmax output dimension')
        parser.add_argument('--embeddings-dropout', type=float, metavar='D',
                            help='dropout probability on the embeddings')
        parser.add_argument('--prediction-dropout', type=float, metavar='D',
                            help='dropout on the final prediction layer')
        parser.add_argument('--need-attention-weights', action='store_true', default=False,
                            help='return attention scores')


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
        """ Build a new model instance. """
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder_embed_tokens, decoder_embed_tokens = cls.share_embeddings(args, src_dict, tgt_dict)
        encoder = PervasiveEncoder(args, src_dict, encoder_embed_tokens)
        decoder = PervasiveDecoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    def max_decoder_positions(self):
        """ Maximum input length supported by the decoder """
        return self.decoder.max_target_positions 


class PervasiveEncoder(FairseqEncoder):
    """
    Prepare the source embeddings for the decoding grid
    """
    def __init__(self, args,  dictionary, embed_tokens):
        super().__init__(dictionary)
        assert not args.left_pad_source , 'Padding of the source should be on the right \
                (use --left-pad-source False)'
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        
        self.embed_tokens = embed_tokens
        self.embed_scale = embed_dim ** .5
        self.embed_positions = PositionalEmbedding(
            self.max_source_positions,
            embed_dim,
            self.padding_idx,
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None

        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = self.embedding_dropout(x)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            'src_tokens': src_tokens, # B, Ts
            'encoder_out': x,  # B, Ts, C
            'encoder_padding_mask': encoder_padding_mask  # B, Ts
        }

    def max_positions(self):
        """ Maximum input length supported by the encoder. """
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def slice_encoder_out(self, encoder_out, context_size):
        """ Reorder encoder output according to *new_order*.  """
        sliced_encoder = {}
        if encoder_out['encoder_out'] is not None:
            sliced_encoder['encoder_out'] = encoder_out['encoder_out'].clone()[:, :context_size]
        else:
            sliced_encoder['encoder_out'] = None
        if encoder_out['encoder_padding_mask'] is not None:
            sliced_encoder['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].clone()[:, :context_size]
        else:
            sliced_encoder['encoder_padding_mask'] = None
        return sliced_encoder

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(0, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


class PervasiveDecoder(FairseqIncrementalDecoder):
    """ Pervasive Attention Model """

    def __init__(self, args,  dictionary, embed_tokens):
        super().__init__(dictionary)
        assert not args.left_pad_target , 'Padding of the target should be on the right'
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.decoder_dim = args.decoder_embed_dim
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = embed_dim ** .5
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            embed_dim,
            self.padding_idx,
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None

        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        self.input_channels = args.encoder_embed_dim + args.decoder_embed_dim
        self.output_dim = args.output_dim

        self.net = build_convnet(args, num_features=self.input_channels)
        self.output_channels = self.net.output_channels

        assert args.aggregator in ['avg', 'max', 'gated-max', 'gated-max2', 'attn',  # offline
                                   'path-max', 'path-gated-max', 'path-attn', # wait-k online
                                   'grid-max', 'grid-gated-max', 'grid-attn', # grid online
                                  ]
        self.aggregator = build_aggregator(args, num_features=self.output_channels)

        if not self.output_dim == self.output_channels or not args.skip_output_mapping:
            self.projection = Linear(
                self.output_channels,
                self.output_dim,
                dropout=args.prediction_dropout
            )

        else:
            self.projection = None
        self.prediction_dropout = nn.Dropout(args.prediction_dropout)
        if self.share_input_output_embed:
            self.prediction = Linear(
                self.decoder_dim,
                len(dictionary)
            )
            self.prediction.weight = self.embed_tokens.weight
        else:
            self.prediction = Linear(
                self.output_dim,
                len(dictionary)
            )
        self.need_attention_weights = args.need_attention_weights

    def forward(
        self, 
        prev_output_tokens, 
        encoder_out,
        incremental_state=None,
        context_size=None,
        cache_decoder=True, 
        **kwargs
    ):
        # source embeddings
        src_emb = encoder_out['encoder_out']
        Ts = src_emb.size(1)
        if context_size is not None:
            if context_size < Ts:
                # src_emb[:, context_size:] = 0
                src_emb = src_emb.clone()[:, :context_size]
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state if cache_decoder else None,
        ) if self.embed_positions is not None else None

        if incremental_state is not None and cache_decoder:
            # embed the last target token
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            
        decoder_mask = prev_output_tokens.eq(self.padding_idx)
        if not decoder_mask.any():
            decoder_mask = None

        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        # pass through convolutional layers
        encoder_mask = encoder_out['encoder_padding_mask']
        x = self.net(
            x, 
            decoder_mask=decoder_mask,
            encoder_mask=encoder_mask,
            incremental_state=incremental_state if cache_decoder else None
        )  # B, Tt, Ts, C

        if incremental_state is not None:
            x, attn = self.aggregator.one_step(x)
        else:
            x, attn = self.aggregator(x, need_attention_weights=self.need_attention_weights)
        x = self.projection(x) if self.projection is not None else x  # B, Tt, C
        x = self.prediction_dropout(x)

        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B, Tt, V
        return x, attn


def _expand(tensor, dim, reps):
    tensor = tensor.unsqueeze(dim)
    shape = tuple(reps if i == dim else -1 for i in range(tensor.dim()))
    return tensor.expand(shape)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0., bias=True):
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(m.weight,
                    mean=0,
                    std=((1 - dropout) / in_features)) ** .5
    nn.init.constant_(m.bias, 0)
    return m


@register_model_architecture('pervasive', 'pervasive')
def base_architecture(args):
    args.skip_output_mapping = getattr(args, 'skip_output_mapping', False)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False
    )
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)

    args.output_dim = getattr(args, 'output_dim', args.decoder_embed_dim)
    args.bottleneck = getattr(
        args, 'bottleneck', 
        (args.encoder_embed_dim + args.decoder_embed_dim) // args.divide_channels
    )
    args.conv_groups = getattr(args, 'conv_groups', args.bottleneck)
    # FIXME
    if args.conv_groups is None:
        args.conv_groups = args.bottleneck
    args.prediction_dropout = getattr(args, 'prediction_dropout', 0.2)
    args.embeddings_dropout = getattr(args, 'embeddings_dropout', 0.2)
