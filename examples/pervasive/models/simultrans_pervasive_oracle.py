""" oracle_attn2d """

import contextlib
import torch
import torch.nn as nn
from fairseq import utils

from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder, FairseqIncrementalDecoder, 
    register_model, register_model_architecture,
)

from examples.pervasive.modules import (
    build_convnet, build_aggregator, PAController
)

from fairseq.modules import (
    PositionalEmbedding,
)

from .pervasive_attention import PervasiveModel, PervasiveEncoder, PervasiveDecoder


@contextlib.contextmanager
def eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


@register_model('simultrans_pervasive_oracle')
class SimultransPervasiveOracleModel(PervasiveModel):

    def __init__(self, encoder, decoder, controller):
        super().__init__(encoder, decoder)
        self.controller = controller

    def forward(self, target, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        logits, grid, decoder_mask = self.decoder(prev_output_tokens, encoder_out)
        if self.controller.remove_writer_dropout:
            with torch.no_grad():
                with eval(self.encoder) and eval(self.decoder):
                    ctrl_logits = self.decoder(
                        prev_output_tokens,
                        self.encoder(src_tokens, src_lengths)
                    )[0]
        else:
            ctrl_logits = logits

        controller_out = self.controller(
           sample={'src_tokens': src_tokens, 
                   'prev_output_tokens': prev_output_tokens, 
                   'target': target
                  },
            encoder_out=encoder_out, 
            decoder_out=(ctrl_logits, grid, decoder_mask),
        )
        return logits, controller_out

    def decide(self, prev_output_tokens, encoder_out, context_size):
        if self.controller.share_embeddings:
            writing_grid = self.decoder.writing_grid(prev_output_tokens,
                                                     encoder_out, 
                                                    context_size)
        else:
            writing_grid = None
        src_tokens = encoder_out['src_tokens']
        if context_size is not None:
            src_tokens = src_tokens[:, :context_size]
        return self.controller.decide(src_tokens, prev_output_tokens, writing_grid)

    @staticmethod
    def add_args(parser):
        PervasiveModel.add_args(parser)
        PAController.add_args(parser)

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
        decoder = PervasiveDynamicDecoder(args, tgt_dict, decoder_embed_tokens)
        controller = PAController(args, src_dict, tgt_dict)
        return cls(encoder, decoder, controller)


class PervasiveDynamicDecoder(PervasiveDecoder):
    """ Pervasive Attention Model """

    def __init__(self, args,  dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
    def writing_grid(self, prev_output_tokens, encoder_out, context_size=None):
        src_emb = encoder_out['encoder_out']
        # source embeddings
        if context_size is not None:
            src_emb = src_emb[:, :context_size]
        # target embeddings:
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=None,
        ) if self.embed_positions is not None else None

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
        grid = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        return grid

    def forward(
        self, 
        prev_output_tokens, 
        encoder_out, 
        incremental_state=None, 
        context_size=None, 
        cache_decoder=False, 
        **kwargs
    ):
        """
        Returns the convolved grid and the grid itself (embeddings).
        """
        # source embeddings
        src_emb = encoder_out['encoder_out']
        Ts = src_emb.size(1)
        if context_size is not None:
            if context_size < Ts:
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
        grid = torch.cat((src_emb, tgt_emb), dim=3)   # B, Tt, Ts, C=ds+dt
        # pass through convolutional layers
        encoder_mask = encoder_out['encoder_padding_mask']
        x = self.net(
            grid, 
            decoder_mask=decoder_mask,
            encoder_mask=encoder_mask,
            incremental_state=incremental_state if cache_decoder else None
        )  # B, Tt, Ts, C

        if incremental_state is not None:
            # Keep only the last step:
            x = x [:, -1:]
            x, _ = x.max(dim=2)  # B, Tt, C  #FIXME one_step
            x = self.projection(x) if self.projection is not None else x  # B, ..., C
            x = self.prediction_dropout(x)
            x = self.prediction(x)  # B, ..., V
            return x, None

        # Training
        x, _ = self.aggregator(x)  # B, Tt, Ts, C
        x = self.projection(x) if self.projection is not None else x  # B, ..., C
        x = self.prediction_dropout(x)

        # multiply by embedding matrix to generate distribution
        x = self.prediction(x)  # B, ..., V
        return x, grid, decoder_mask

        
@register_model_architecture('simultrans_pervasive_oracle', 'simultrans_pervasive_oracle')
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
    args.prediction_dropout = getattr(args, 'prediction_dropout', 0.2)
    args.embeddings_dropout = getattr(args, 'embeddings_dropout', 0.2)
    args.control_gate_dropout = getattr(args, 'control_gate_dropout', 0.)
    

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
    nn.init.normal_(m.weight, mean=0,
                    std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return m



