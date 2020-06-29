# Copyright (c) 2017-present, Facebook, Inc.  # All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import torch

from fairseq import options
from fairseq.data import Dictionary, data_utils
from . import register_task
from .translation import TranslationTask


@register_task('dynamic_ll_simultaneous_translation')
class DynamicLLSimultaneousTranslationTask(TranslationTask):
    """
    """
    def adapt_state(self, state_dict, model):
        # Add random weights for the gate:
        model_state = model.state_dict()
        for k in model_state:
            if k not in state_dict:
                if self.args.copy_embeddings and 'embed' in k:
                    print('Copying embeddings:', self.args.copy_embeddings)
                    print('k:', k, '>>>', k.replace('ctrl_', ''))
                    state_dict[k] = state_dict[k.replace('ctrl_', '')]
                else:
                    print('Keeping Asis:', k)
                    state_dict[k] = model_state[k]
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        # Training:
        parser.add_argument('--copy-embeddings', action='store_true')
        parser.add_argument('--shift', default=1, type=int)
        parser.add_argument('--delta', default=1, type=int)
        parser.add_argument('--catchup', default=1, type=int)
        parser.add_argument('--policy', default='eos', type=str)
        parser.add_argument('--align-beams', action='store_true')
        parser.add_argument('--patience', default=3, type=int)
        parser.add_argument('--write-threshold', default=0.5, type=float)
        parser.add_argument('--above-diagonal', action='store_true')
        parser.add_argument('--force-end-reading', action='store_true')
        parser.add_argument('--end-scale', type=float, default=1)
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # args.left_pad_source = options.eval_bool(args.left_pad_source)
        # args.left_pad_target = options.eval_bool(args.left_pad_target)
        args.left_pad_source = False
        args.left_pad_target = False
        args.prepend_bos_to_source = options.eval_bool(args.prepend_bos_to_source)
        args.remove_eos_from_source = options.eval_bool(args.remove_eos_from_source)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def build_generator(self, args):
        if args.policy == 'path':
            from fairseq.simultaneous_path_sequence_generator import SimultaneousPathSequenceGenerator
            return SimultaneousPathSequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                shift=args.shift,
                delta=args.delta,
                catchup=args.catchup,
                update_context=args.update_context,
            )
        if args.policy == 'token':
            from fairseq.simultaneous_eos_sequence_generator import SimultaneousEosSequenceGenerator
            return SimultaneousEosSequenceGenerator(
                self.target_dictionary, beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=3,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                shift=args.shift,
                patience=args.patience,
            )

        if args.policy == 'sig':
            from fairseq.simultaneous_sig_sequence_generator import SimultaneousSigSequenceGenerator
            return SimultaneousSigSequenceGenerator(
                self.target_dictionary, beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=3,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                shift=args.shift,
                patience=args.patience
            )

        if args.policy == 'diag':
            from fairseq.simultaneous_diag_sequence_generator import SimultaneousDiagSequenceGenerator
            return SimultaneousDiagSequenceGenerator(
                self.target_dictionary, beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=3,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                shift=args.shift,
                patience=args.patience,
                align_beams=args.align_beams
            )
        if args.policy == 'dynamic':
            from fairseq.dynamic_simultaneous_sequence_generator import DynamicSimultaneousSequenceGenerator
            return DynamicSimultaneousSequenceGenerator(
                self.target_dictionary, beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=3,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                shift=args.shift,
                patience=args.patience,
                write_threshold=args.write_threshold,
                )
        if args.policy == 'oracle':
            from fairseq.oracle_path_sequence_generator import OraclePathSequenceGenerator
            return OraclePathSequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                without_caching=args.without_caching,
                alignment_index=args.align_index,
                pick_alignment=args.pick_alignment,
                oracle=args.path_oracle,
                oracle_rank=args.path_oracle_rank,
                oracle_tol=args.path_oracle_tol,
                look_ahead=args.path_oracle_lookahead,
                oracle_waitk=args.path_oracle_waitk,
                oracle_width=args.path_oracle_width,
            )


    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, **kwargs):
        """
        Training iteration
        """
        model.train()
        # forward pass
        emissions, observations, controls, gamma, read_labels, write_labels = model(**sample['net_input'], target=sample['target'])
        loss, sample_size, logging_output = criterion(sample, emissions, controls, gamma, read_labels, write_labels)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            # forward pass
            emissions, observations, controls, gamma, read_labels, write_labels = model(**sample['net_input'], target=sample['target'])
            loss, sample_size, logging_output = criterion(sample, emissions, controls, gamma, read_labels, write_labels)
        return loss, sample_size, logging_output

