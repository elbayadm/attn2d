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


@register_task('search_simultaneous_translation')
class SearchSimultaneousTranslationTask(TranslationTask):
    """
    """
    def adapt_state(self, state_dict, model):
        # Add random weights for the gate:
        model_state = model.state_dict()
        for k in model_state:
            if k not in state_dict:
                print('Keeping Asis:', k)
                state_dict[k] = model_state[k]
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--shift', default=5, type=int)
        parser.add_argument('--delta', default=1, type=int)
        parser.add_argument('--catchup', default=1, type=int)
        parser.add_argument('--policy', default='eos', type=str)
        parser.add_argument('--align-beams', action='store_true')
        parser.add_argument('--patience', default=3, type=int)
        parser.add_argument('--read-threshold', default=0.5, type=float)
        parser.add_argument('--update-context', action='store_true')
        parser.add_argument('--force-end-reading', action='store_true')
        parser.add_argument('--end-scale', type=float, default=1)

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
                force_end_reading=args.force_end_reading,
                end_scale=args.end_scale,
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
                read_threshold=args.read_threshold
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
                read_threshold=args.read_threshold,
                align_beams=args.align_beams
            )



    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, **kwargs):
        """
        """
        model.train()
        target = sample['target']
        target_lengths = sample['tgt_lengths']

        # forward pass
        likelihoods, read_signals = model(**sample['net_input'], target=target)
        loss, sample_size, logging_output = criterion(sample, likelihoods, read_signals)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            target = sample['target']
            target_lengths = sample['tgt_lengths']
            # forward pass
            likelihoods, read_signals = model(**sample['net_input'], target=target)
            loss, sample_size, logging_output = criterion(sample, likelihoods, read_signals)
        return loss, sample_size, logging_output

