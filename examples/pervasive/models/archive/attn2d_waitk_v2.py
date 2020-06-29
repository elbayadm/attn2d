import math
import torch
import torch.nn as nn
from fairseq import utils

# Only for Grid


from . import (
    FairseqModel, FairseqEncoder, FairseqIncrementalDecoder,
    register_model, register_model_architecture,
)

from fairseq.modules import (
    DenseNetLN, DenseNetBN, DenseNetNoNorm, DenseNetPONO, DenseNetFFN, 
    DenseNetFFNPONO, DenseNetPONOCascade, DenseNetPONOKmax, DenseNetCascade,
    ResNet, ResNet2, ResNet3, ResNet4, ResNet5, ResNet6,
    DilatedResnet, DilatedResnet2, ExpandingResNet,
    ResNetReNorm, FavResNet, ResNetAddUpNoNorm2,
    ResNetAddUpNoNorm4, ResNetAddUpNoNorm,
    ResNetAddUp, ResNetAddUp2, ResNetAddUp3,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    GridMAX, GridGatedMAX, GridATTN, GridMAX2,
    PathMAX, PathGatedMAX, PathATTN, PathCell, PathMAXFull,
)


@register_model('attn2d_waitk_v2')
class Attn2dCPModel(FairseqModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def get_lenx(self, encoder_out):
        return encoder_out['encoder_out'].size(1)

    def forward_stats(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.decoder.forward_stats(prev_output_tokens, encoder_out)

    @staticmethod
    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        """ Embeddings """
        parser.add_argument('--skip-output-mapping', action='store_true',
                            help='remove the final mapping if equal dimension')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--add-positional-embeddings', default=False, action='store_true',
                            help='if set, enables positional embeddings')
        parser.add_argument('--learned-pos', action='store_true',
                            help='use learned positional embeddings')

        parser.add_argument('--aggregation', type=str, default='max')

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
        parser.add_argument('--conv-bias', action='store_true',
                            help='Add bias in the first conv')
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
        parser.add_argument('--layer-type', type=str, metavar='STR',
                            help='Type of Residual layer either standard (default) or Macaron')

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
        parser.add_argument('--num-cascade-layers', type=int, 
                            help='number of cascading layers')

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
        parser.add_argument('--nonzero-padding', action='store_true',
                            help='Do not zero out padding positions in the conv activations')
        parser.add_argument('--agg-zero-padding', action='store_true',
                            help='Do not zero out padding positions before aggregation')
        parser.add_argument('--num-heads', type=int)
        parser.add_argument('--need-attention-weights', action='store_true')
        # Decoding path:
        parser.add_argument('--waitk-policy', type=str, default='area',
                            help='The type of fixed policy with fixed mnaginals')
        parser.add_argument('--waitk', type=int, default=3, help='Fixed policy shift')
        parser.add_argument('--lower-diag', type=int, default=5)
        parser.add_argument('--upper-diag', type=int, default=5)


    def log_tensorboard(self, writer, iter):
        pass
        # for name, param in self.named_parameters():
            # # writer.add_histogram(name,  param.clone().cpu().data.numpy(), iter)
            # if 'mconv' in name and 'weight' in name:
                # C, D, H, W = param.size()
                # im = param.clone().view(C*D, 1, H, W)
                # im = vutils.make_grid(im,  normalize=True,  scale_each=True)
                # writer.add_image(name, im, iter)
    # writer.add_graph(,  dummy_input,  True)

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
        # return 1024


class Attn2dEncoder(FairseqEncoder):
    def __init__(self, args,  dictionary, embed_tokens):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            self.max_source_positions,
            embed_dim, self.padding_idx,
            left_pad=args.left_pad_source,  # False
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None

        self.ln = lambda x: x
        if args.embeddings_ln:
            self.ln = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = self.ln(x)
        x = self.embedding_dropout(x)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            'encoder_out': x,  # B, Ts, C
            'encoder_padding_mask': encoder_padding_mask  #B, Ts
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

    def __init__(self, args,  dictionary, embed_tokens):
        super().__init__(dictionary)
        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.decoder_dim = args.decoder_embed_dim
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions,
            embed_dim, self.padding_idx,
            left_pad=args.left_pad_target,  # False
            learned=args.learned_pos,
        ) if args.add_positional_embeddings else None
        self.ln = lambda x: x
        if args.embeddings_ln:
            self.ln = nn.LayerNorm(embed_dim, elementwise_affine=True)

        self.embedding_dropout = nn.Dropout(args.embeddings_dropout)
        self.input_dropout = nn.Dropout(args.input_dropout)
        self.input_channels = args.encoder_embed_dim + args.decoder_embed_dim
        self.output_dim = args.output_dim

        if args.network == 'resnet':
            self.net = ResNet(self.input_channels, args)
        elif args.network == 'resnet2':
            self.net = ResNet2(self.input_channels, args)
        elif args.network == 'dilated_resnet':
            self.net = DilatedResnet(self.input_channels, args)
        elif args.network == 'dilated_resnet2':
            self.net = DilatedResnet2(self.input_channels, args)
        elif args.network == 'expanding_resnet':
            self.net = ExpandingResNet(self.input_channels, args)


        elif args.network == 'fav_resnet':
            self.net = FavResNet(self.input_channels, args)

        elif args.network == 'resnet3':
            self.net = ResNet3(self.input_channels, args)
        elif args.network == 'resnet4':
            self.net = ResNet4(self.input_channels, args)
        elif args.network == 'resnet5':
            self.net = ResNet5(self.input_channels, args)
        elif args.network == 'resnet6':
            self.net = ResNet6(self.input_channels, args)
        elif args.network == 'resnet_renorm':
            self.net = ResNetReNorm(self.input_channels, args)
        elif args.network == 'resnet_addup':
            self.net = ResNetAddUp(self.input_channels, args)
        elif args.network == 'resnet_addup2':
            self.net = ResNetAddUp2(self.input_channels, args)
        elif args.network == 'resnet_addup3':
            self.net = ResNetAddUp3(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm':
            self.net = ResNetAddUpNoNorm(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm2':
            self.net = ResNetAddUpNoNorm2(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm2_rev':
            self.net = ResNetAddUpNoNorm2Rev(self.input_channels, args)

        elif args.network == 'resnet_addup_nonorm2_wbias':
            self.net = BiasResNetAddUpNoNorm2(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm2_gated':
            self.net = ResNetAddUpNoNorm2Gated(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm2_all':
            self.net = ResNetAddUpNoNorm2All(self.input_channels, args)

        elif args.network == 'resnet_addup_nonorm2_gated_noffn':
            self.net = ResNetAddUpNoNorm2GatedNoFFN(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm3':
            self.net = ResNetAddUpNoNorm3(self.input_channels, args)
        elif args.network == 'resnet_addup_nonorm4':
            self.net = ResNetAddUpNoNorm4(self.input_channels, args)
        elif args.network == 'densenet_ln':
            self.net = DenseNetLN(self.input_channels, args)
        elif args.network == 'densenet_ffn':
            self.net = DenseNetFFN(self.input_channels, args)
        elif args.network == 'densenet_ffn_pono':
            self.net = DenseNetFFNPONO(self.input_channels, args)
        elif args.network == 'densenet_pono':
            self.net = DenseNetPONO(self.input_channels, args)
        elif args.network == 'densenet_pono_cascade':
            self.net = DenseNetPONOCascade(self.input_channels, args)
        elif args.network == 'densenet_cascade':
            self.net = DenseNetCascade(self.input_channels, args)

        elif args.network == 'densenet_pono_kmax':
            self.net = DenseNetPONOKmax(self.input_channels, args)
        elif args.network == 'densenet_bn':
            self.net = DenseNetBN(self.input_channels, args)
        elif args.network == 'densenet_nonorm':
            self.net = DenseNetNoNorm(self.input_channels, args)

        else:
            raise ValueError('Unknown architecture %s' % args.network)

        self.policy = args.waitk_policy
        self.waitk = args.waitk

        self.output_channels = self.net.output_channels
        if args.waitk_policy == 'area':
            if args.aggregation == 'max':
                self.aggregator = GridMAX(self.output_channels)
            elif args.aggregation == 'max2':
                self.aggregator = GridMAX2(self.output_channels)
            elif args.aggregation == 'gated_max':
                self.aggregator = GridGatedMAX(self.output_channels)
            elif args.aggregation == 'attn':
                self.aggregator = GridATTN(self.output_channels)
            else:
                raise ValueError('Unknown aggregation %s' % args.aggregation)
        elif args.waitk_policy == 'path':
            if args.aggregation == 'max':
                self.aggregator = PathMAX(self.output_channels, self.waitk)
            elif args.aggregation == 'gated_max':
                self.aggregator = PathGatedMAX(self.output_channels, self.waitk)
            elif args.aggregation == 'attn':
                self.aggregator = PathATTN(self.output_channels, self.waitk)
            elif args.aggregation == 'cell':
                self.aggregator = PathCell(self.output_channels, self.waitk)
            elif args.aggregation == 'full':
                self.aggregator = PathMAXFull(self.output_channels, self.waitk)

            else:
                raise ValueError('Unknown aggregation %s' % args.aggregation)
        else:
            raise ValueError('Unknown policy %s' % args.waitk_policy)


        print('Decoder dim:', self.decoder_dim)
        print('The ConvNet output channels:', self.output_channels)
        print('Required output dim:', self.output_dim)
        
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

    def forward_stats(self, prev_output_tokens, encoder_out):
        # source embeddings
        src_emb = encoder_out['encoder_out'].clone()  # B, Ts, ds 
        Ts = src_emb.size(1)
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
        ) if self.embed_positions is not None else None

        decoder_mask = prev_output_tokens.eq(self.padding_idx)
        if not decoder_mask.any():
            decoder_mask = None

        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions
        tgt_emb = self.ln(tgt_emb)

        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        # pass through dense convolutional layers
        encoder_mask = encoder_out['encoder_padding_mask']
        stats = self.net.forward_stats(
            x, 
            decoder_mask=decoder_mask,
            encoder_mask=encoder_mask,
        )  # B, Tt, Ts, C
        return stats, x.size(0)*x.size(1)*x.size(2)

    def forward(self, prev_output_tokens, encoder_out,
                incremental_state=None,
                context_size=None,
                cache_decoder=True, **kwargs):
        torch.set_printoptions(precision=2, threshold=5000)
        # source embeddings
        src_emb = encoder_out['encoder_out'].clone()  # B, Ts, ds 
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

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        encoder_mask = encoder_out['encoder_padding_mask']
        x = self.net(
            x, 
            decoder_mask=decoder_mask,
            encoder_mask=encoder_mask,
            incremental_state=incremental_state if cache_decoder else None
        )  # B, Tt, Ts, C

        if incremental_state is not None:
            # Keep only the last step:
            # x = x[:, -1, :context_size]
            # upto = min(context_size, src_length)
            # x = x[:, :, :upto]
            x, attn = self.aggregator.one_step(
                x, 
                need_attention_weights=self.need_attention_weights
            )
        else:
            x, attn = self.aggregator(x, need_attention_weights=self.need_attention_weights)
        # x in B, Tt, Ts, C
        x = self.projection(x) if self.projection is not None else x 
        x = self.prediction_dropout(x)
        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B, Tt,Ts,V
        return x, attn

    def get_wue_align(self, prev_output_tokens,
                      encoder_out, incremental_state=None):
        # source embeddings
        src_emb = encoder_out['encoder_out']  # B, Ts, ds 
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
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = self.net(x, incremental_state)  # B, Tt, Ts, C
        x, indices = x.max(dim=2)  # B, Tt, C
        # only works for N=1
        counts = [torch.histc(indices[:, i], bins=src_length, min=0, max=src_length-1) for i in range(tgt_length)]
        counts = [c.float()/torch.sum(c) for c in counts]
        align = torch.stack(counts, dim=0).unsqueeze(0)  # 1, Tt, Ts
        return [align]

    def forward_one(self, prev_output_tokens, encoder_out, context_size,
                    incremental_state=None, **kwargs):
        # Truncate the encoder outputs:
        encoder_out_truncated = {'encoder_out': encoder_out['encoder_out'].clone()[:,:context_size]}

        x, attn = self.forward(prev_output_tokens, encoder_out_truncated,
                            incremental_state)
        return x, {'attn': attn}

    def forward_one_with_update(self, prev_output_tokens, encoder_out, context_size,
                                incremental_state=None, **kwargs):
        # Truncate the encoder outputs:
        encoder_out_truncated = {'encoder_out': encoder_out['encoder_out'].clone()[:,:context_size]}

        # source embeddings
        src_emb = encoder_out_truncated['encoder_out']  # B, Ts, ds 
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # embed the last target token
            prev_output_tokens = prev_output_tokens#[:, -1:]
            if positions is not None:
                positions = positions#[:, -1:]
            
        # Build the full grid
        tgt_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            tgt_emb += positions

        tgt_emb = self.ln(tgt_emb)
        tgt_emb = self.embedding_dropout(tgt_emb)
                
        src_length = src_emb.size(1)
        tgt_length = tgt_emb.size(1)

        # build 2d "image" of embeddings
        src_emb = _expand(src_emb, 1, tgt_length)  # B, Tt, Ts, ds
        tgt_emb = _expand(tgt_emb, 2, src_length)  # B, Tt, Ts, dt
        x = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        x = self.input_dropout(x)
        # pass through dense convolutional layers
        x = self.net(x)  # B, Tt, Ts, C
        # Only the last step:
        x = x[:, -1:]
        # aggregate predictions and project into embedding space
        x, _ = x.max(dim=2)  # B, Tt, C
        x = self.projection(x) if self.projection is not None else x  # B, Tt, C
        x = self.prediction_dropout(x)

        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B, Tt, V
        return x, {'attn': None}


def _expand(tensor, dim, reps):
    tensor = tensor.unsqueeze(dim)
    shape = tuple(reps if i == dim else -1 for i in range(tensor.dim()))
    return tensor.expand(shape)


def PositionalEmbedding(num_embeddings, embedding_dim,
                        padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1,
                                       embedding_dim, padding_idx, left_pad,
                                       clamp=True)
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


@register_model_architecture('attn2d_waitk_v2', 'attn2d_waitk_v2')
def base_architecture(args):
    args.memory_efficient = getattr(args, 'memory_efficient', False)
    args.nonzero_padding = getattr(args, 'nonzero_padding', False)
    args.agg_zero_padding = getattr(args, 'agg_zero_padding', False)

    args.conv_bias = getattr(args, 'conv_bias', False)
    args.aggregation = getattr(args, 'aggregation', 'max')


    args.skip_output_mapping = getattr(args, 'skip_output_mapping', False)

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
    args.divide_channels = getattr(args, 'divide_channels', 2)
    args.reduce_dim = getattr(args, 'reduce_dim',
                              (args.encoder_embed_dim + args.decoder_embed_dim) // args.divide_channels)
    args.conv_groups = getattr(args, 'conv_groups', args.reduce_dim)
    args.conv_stride = getattr(args, 'conv_stride', 1)
    args.source_dilation = getattr(args, 'source_dilation', 1)
    args.target_dilation = getattr(args, 'target_dilation', 1)
    args.maintain_resolution = getattr(args, 'maintain_resolution', 1)

    args.add_positional_emnbeddings = getattr(args, 'add_positional_embeddings', False)
    args.learned_pos = getattr(args, 'learned_pos', False)
    args.embeddings_ln = getattr(args, 'embeddings_ln', False)

    args.input_dropout = getattr(args, 'input_dropout', 0.2)
    args.convolution_dropout = getattr(args, 'convolution_dropout', 0.2)

    args.network = getattr(args, 'network', 'resnet')
    args.layer_type = getattr(args, 'layer_type', 'standard')
    args.num_heads = getattr(args, 'num_heads', 1)
    args.growth_rate = getattr(args, 'growth_rate', 32)
    args.kernel_size = getattr(args, 'kernel_size', 3)
    args.bn_size = getattr(args, 'bn_size', 4)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.num_cascade_layers = getattr(args, 'num_cascade_layers', 2)


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
    args.double_masked = getattr(args, 'double_masked', True) # By default
    args.need_attention_weights = getattr(args, 'need_attention_weights', False)
    args.waitk_policy = getattr(args, 'waitk_policy', 'area')
    args.waitk = getattr(args, 'waitk', 1)
