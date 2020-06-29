import os
import torch

from fairseq import options
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task('dynamic_pervasive_simultaneous_translation')
class DynamicPervasiveSimultaneousTranslationTask(TranslationTask):
    def adapt_state(self, state_dict, model):
        # Add random weights for the gate:
        model_state = model.state_dict()
        for k in model_state:
            if k not in state_dict:
                if self.args.copy_embeddings and 'embed' in k:
                    assert model.controller.src_embed_tokens.embedding_dim == model.encoder.embed_tokens.embedding_dim
                    assert model.controller.tgt_embed_tokens.embedding_dim == model.decoder.embed_tokens.embedding_dim
                    if 'controller.src_embed' in k:
                        copy_k = k.replace('controller.src_', 'encoder.')
                    elif 'controller.tgt_embed' in k:
                        copy_k = k.replace('controller.tgt_', 'decoder.')
                    print('Copying ', copy_k, 'into', k)
                    state_dict[k] = state_dict[copy_k]
                elif self.args.copy_network and 'controller.net' in k:
                    copy_k = k.replace('controller', 'decoder')
                    print('Copying ', copy_k, state_dict[copy_k].size(), 'into', k, model_state[k].size())
                    print('')
                    assert model_state[k].size() == state_dict[copy_k].size()
                    state_dict[k] = state_dict[copy_k]
                else:
                    print('Keeping Asis:', k)
                    state_dict[k] = model_state[k]
        model.load_state_dict(state_dict, strict=True)

    
    def build_model(self, args):
        model = super().build_model(args)
        if args.pretrained is not None: # load pretrained model:
            if not os.path.exists(args.pretrained):
                raise ValueError('Could not load pretrained weights \
                                 - from {}'.format(args.pretrained))
            from torch.serialization import default_restore_location
            saved_state = torch.load(
                args.pretrained, 
                map_location=lambda s, l: default_restore_location(s, 'cpu')
            )
            self.adapt_state(saved_state['model'], model)

        return model

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        # Training:
        parser.add_argument('--pretrained', default=None, type=str)
        parser.add_argument('--copy-embeddings', action='store_true')
        parser.add_argument('--copy-network', action='store_true')
        parser.add_argument('--shift', default=1, type=int)
        parser.add_argument('--policy', default='eos', type=str)
        parser.add_argument('--write-threshold', default=0.5, type=float)
        parser.add_argument('--align-index', default=1, type=int)
        parser.add_argument('--pick-alignment', default='index', type=str)
        parser.add_argument('--path-oracle', default='alignment', type=str)
        parser.add_argument('--path-oracle-rank', default=50, type=int)
        parser.add_argument('--path-oracle-tol', default=0.1, type=float)
        parser.add_argument('--path-oracle-lookahead', default=0, type=int)
        parser.add_argument('--path-oracle-waitk', default=7, type=int)
        parser.add_argument('--path-oracle-width', default=3, type=int)
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        assert not args.left_pad_source, 'Source should padded to the rigth'
        assert not args.left_pad_target, 'Target should be padded to the right'
    
    def train_step(self, sample, model, 
                   criterion, optimizer, 
                   update_num, ignore_grad=False):
        """ Training iteration """
        model.train()
        # forward pass
        logits, controller_out = model(**sample['net_input'], target=sample['target'])
        loss, sample_size, logging_output = criterion(sample, logits, controller_out)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            # forward pass
            logits, controller_out = model(**sample['net_input'], target=sample['target'])
            loss, sample_size, logging_output = criterion(sample, logits, controller_out)
        return loss, sample_size, logging_output

    def build_generator(self, models, args):
        if args.policy == 'dynamic':
            from examples.pervasive.generators.dynamic_simultaneous_sequence_generator import DynamicPervasiveSequenceGenerator
            return DynamicPervasiveSequenceGenerator(
                models,
                self.target_dictionary, 
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=3,
                match_source_len=args.match_source_len,
                shift=args.shift,
                write_threshold=args.write_threshold,
                )
        # if args.policy == 'oracle':
            # from fairseq.oracle_path_sequence_generator import OraclePathSequenceGenerator
            # return OraclePathSequenceGenerator(
                # self.target_dictionary,
                # beam_size=args.beam,
                # max_len_a=args.max_len_a,
                # max_len_b=args.max_len_b,
                # min_len=args.min_len,
                # stop_early=(not args.no_early_stop),
                # normalize_scores=(not args.unnormalized),
                # len_penalty=args.lenpen,
                # unk_penalty=args.unkpen,
                # sampling=args.sampling,
                # sampling_topk=args.sampling_topk,
                # sampling_temperature=args.sampling_temperature,
                # diverse_beam_groups=args.diverse_beam_groups,
                # diverse_beam_strength=args.diverse_beam_strength,
                # match_source_len=args.match_source_len,
                # no_repeat_ngram_size=args.no_repeat_ngram_size,
                # without_caching=args.without_caching,
                # alignment_index=args.align_index,
                # pick_alignment=args.pick_alignment,
                # oracle=args.path_oracle,
                # oracle_rank=args.path_oracle_rank,
                # oracle_tol=args.path_oracle_tol,
                # look_ahead=args.path_oracle_lookahead,
                # oracle_waitk=args.path_oracle_waitk,
                # oracle_width=args.path_oracle_width,
            # )

