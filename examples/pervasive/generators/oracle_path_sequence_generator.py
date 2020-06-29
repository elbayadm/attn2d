# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys
import math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder


class OraclePathSequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        stop_early=True,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        without_caching=False,
        alignment_index=1,
        pick_alignment='index',
        oracle='alignment',
        oracle_rank=50,
        oracle_tol=0,
        oracle_waitk=7,
        oracle_width=3,
        look_ahead=0,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            stop_early (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_temperature (float, optional): temperature for sampling,
                where values >1.0 produces more uniform sampling and values
                <1.0 produces sharper sampling (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.without_caching = without_caching
        self.alignment_index = alignment_index
        self.pick_alignment = pick_alignment
        self.oracle = oracle
        self.oracle_rank = oracle_rank
        self.oracle_tol = oracle_tol
        self.look_ahead = look_ahead
        self.oracle_waitk = oracle_waitk
        self.oracle_width = oracle_width
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'

        self.search = search.SimBeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        model = EnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

                    
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        dup_src_lengths = src_lengths.clone()
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # encoder outs evaluated once:
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
        dup_src_lengths = dup_src_lengths.index_select(0, new_order)

        # Infer the path in teacher-forcing mode:
        torch.set_printoptions(precision=1, threshold=5000)
        oracle_context, oracle_meta = model.get_oracle_contexts(sample['net_input'],
                                                                sample['target'], 
                                                                oracle=self.oracle,
                                                                oracle_tol=self.oracle_tol,
                                                               )
        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = bos_token or self.eos
        attn, attn_buf = None, None
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= max_len ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos

            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():
                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    seq = tokens_clone[i][tokens_clone[i] != self.eos]
                    # truncate contexts up to source length:
                    ctx_hyp = [
                        min(c,src_lengths[idx]) for c in used_contexts
                    ][:len(tokens_clone[i])-1] # Remove EOS context (src length does not include eos either)

                    hypo = {
                        'tokens': seq,
                        'score': score,
                        'context': ctx_hyp,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }
                    if oracle_meta is not None:
                        hypo['align_meta'] = oracle_meta
                    return hypo

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None

        
        used_contexts = []
        for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                model.reorder_encoder_out(encoder_outs, reorder_state)
                dup_src_lengths = dup_src_lengths.index_select(0, reorder_state)

            # decode t with context shift + [t/catchup] delta
            # Get ctx from oracle 
            # ctx = self.shift + ((step) // self.catchup) * self.delta
            if step < len(oracle_context):
                ctx = oracle_context[step]
            else:
                ctx += 1
            used_contexts.append(ctx)
            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1],
                encoder_outs,
                context_size=ctx,
                without_caching=self.without_caching
            )
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            
            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < max_len:
                self.search.set_src_lengths(src_lengths)
                if self.no_repeat_ngram_size > 0:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, 
                        # prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(tokens[bbsz_idx, step + 2 -
                                                   self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated 
                        # no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx)
                                         for bbsz_idx in range(bsz * beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                    for bbsz_idx in range(bsz * beam_size):
                        lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1)
                    ).view(-1, 1).repeat(1, cand_size)
                    if step > 0:
                        # save cumulative scores for each hypothesis
                        cand_scores.add_(scores[:, step - 1].view(bsz, beam_size).repeat(1, 2))
                    cand_indices = prefix_tokens[:, step].view(-1, 1).repeat(1, cand_size)
                    cand_beams = torch.zeros_like(cand_indices)

                    # handle prefixes of different lengths
                    partial_prefix_mask = prefix_tokens[:, step].eq(self.pad)
                    if partial_prefix_mask.any():
                        partial_scores, partial_indices, partial_beams = self.search.step(
                            step,
                            lprobs.view(bsz, -1, self.vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                        )
                        cand_scores[partial_prefix_mask] = partial_scores[partial_prefix_mask]
                        cand_indices[partial_prefix_mask] = partial_indices[partial_prefix_mask]
                        cand_beams[partial_prefix_mask] = partial_beams[partial_prefix_mask]
                else:
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                    )

            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))
                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos) # * (ctx >= 0.9 * src_lengths).view(bsz, beam_size)
            finalized_sents = set()
            if step >= self.min_len:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        try:
            self.num_heads = models[0].decoder.layers[0].encoder_attn.num_heads
            self.num_blocks = len(models[0].decoder.layers)
        except:
            pass
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        """
        src_tokens :  B, Ts
        read_lenghts: gives thee index of the last tokn to read. B
        """
        # maksk unread tokens:
        return [model.encoder(**encoder_input) for model in self.models]
        
    @torch.no_grad()
    def get_oracle_contexts(self, net_input, target, oracle='likelihood', oracle_tol=0):
        if oracle == 'likelihood':
            return self.get_likelihood_contexts(net_input, target, oracle_tol)
        elif oracle == 'fb':
            return self.get_fb(net_input, target, oracle_tol)
        elif oracle == 'fb2':
            return self.get_fb2(net_input, target, oracle_tol)
        elif oracle == 'fb2_debug':
            return self.get_fb2_debug(net_input, target, oracle_tol)
        elif oracle == 'fb2_diag':
            return self.get_fb2_diag(net_input, target, oracle_tol)
        elif oracle == 'fb2_shift':
            return self.get_fb2_shift(net_input, target, oracle_tol)
        elif oracle == 'fb3':
            return self.get_fb3(net_input, target, oracle_tol)
        elif oracle == 'forward':
            return self.get_forward(net_input, target, oracle_tol)
        elif oracle == 'rank':
            return self.get_rank(net_input, target, oracle_tol)
        else:
            raise ValueError('Unknown oracle', oracle)


    def get_likelihood_contexts(self, net_input, target, tol=0):
        def progressive_max(x):
            T = x.size(1)
            x = F.pad(x, (T-1, 0), 'constant', -1)
            x = F.max_pool1d(x.unsqueeze(1).float(),  # shape into B, C, T
                            T, # kernel size
                            1, # stride
                            0, # padding
                            1, # dilation
                            False, # ceil_mode
                            False, # return indices
                            )
            return x.squeeze(1)  # B, Tt

        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores = scores.view(-1, V).gather(
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)

        # Penalize large contexts:
        indices = torch.arange(
            Ts,
            dtype=scores.dtype,
            device=scores.device
        ) / Ts
        scores = scores - tol * indices.unsqueeze(0).unsqueeze(0)
        best_context = 1 + scores.max(-1)[1]  # B, Tt
        best_context = progressive_max(best_context).type_as(best_context)
        AP = best_context.float().mean(dim=1) / Ts
        print('AP:', ' '.join(map(lambda x: '{:.2f}'.format(x), AP.tolist())))
        print('Monotonic Best contexts:', best_context)
        return best_context[0].tolist(), None  # batch size =1

    def get_forward(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # First column:
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            # First row:
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores

        cs = forward_scores(scores, read_cost)
        # print('Forward scores:')
        # print(cs)
        # Find the lowest scoring path:
        t = Tt
        j = Ts-1
        best_context = []
        while t > 0 and j > 0:
            read_score = cs[t, j-1]
            write_score = cs[t-1, j]
            if write_score <= read_score: # write  # Added equality in favor of writing
                best_context.insert(0, j+1)
                t -= 1
            else:  # read
                # best_context.insert(0, j+1)
                # t -= 1
                j -= 1
        # print('Exiting with', best_context)
        while len(best_context) < Tt:
            best_context.insert(0, 1)
        print('Best context:', best_context)
        return best_context, None

    def get_rank(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = - utils.log_softmax(x, dim=-1)  # NLL: B, Tt, Ts, V
        # Ground truth
        gt = scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts, 1)
        scores = scores.lt(gt).sum(dim=-1)[0].float() / V  # Find the rank and map to [0,1]
        # print('Scores:', scores)
                        
        def backward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores

        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        # print('Accumulated scores:')
        # print(cs)
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None

    def get_fb3(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        confidence = scores.max(dim=-1)[0][0]  # B, Tt, Ts
        scores =  scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]

        # a point if you get it right:
        scores = scores.ne(confidence).int()
        print('Right', scores)
                        
        def backward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores

        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        print('Accumulated scores:')
        print(cs)
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None

    def get_fb2_debug(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        confidence = - scores.max(dim=-1)[0][0]  # B, Tt, Ts
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]
        # Divide by the confidence:
        # print('NLL scores:')
        # print(scores)
        # print('Normalized by the confidence:')
        # scores = scores / confidence
        print(scores)
       
        def backward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(ifwrite, ifread)
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(ifwrite, ifread)
            return path_scores

        # Forward scores:
        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        print('Accumulated scores:')
        print(cs)
        # Find the lowest scoring path:
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None


    def get_fb2_shift(self, net_input, target, read_cost=1, shift=2):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]

                        
        def backward_scores(scores, read_cost):
            # Expected future cost
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # Last column:
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            # Last row:
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            # print('Init backward')
            # print(path_scores)
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Backward:')
            # print(path_scores)
            return path_scores 

        def forward_scores(scores, read_cost, shift=2):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # First column:
            path_scores[1:, 0] = torch.cumsum(scores[:, 1], dim=-1)

            # First row:
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            # print('Init forward')
            # print(path_scores)
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Forward')
            # print(path_scores)
            return path_scores

        # print('NLL scores:')
        # print(scores)
        # Forward scores:
        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        # print('Accumulated scores:')
        # print(cs)
        # Find the lowest scoring path:
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None

    def get_fb2_diag(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]
        # Forbid below diagonal:
        mask = torch.tril(scores.new_ones(Tt, Ts) * 1000, -max(Tt-Ts, 0)-1)  # + infty
        # print('Mask', mask)
        scores = scores + mask

                        
        def backward_scores(scores, read_cost):
            # Expected future cost
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # Last column:
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            # Last row:
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            # print('Init backward')
            # print(path_scores)
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Backward:')
            # print(path_scores)
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # First column:
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            # First row:
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            # print('Init forward')
            # print(path_scores)
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Forward')
            # print(path_scores)
            return path_scores

        # print('NLL scores:')
        # print(scores)
        # Forward scores:
        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        # print('Accumulated scores:')
        # print(cs)
        # Find the lowest scoring path:
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None

    def get_fb2(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]
        # print('NLL scores:')
        # print(scores)

                        
        def backward_scores(scores, read_cost):
            # Expected future cost
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # Last column:
            path_scores[:-1, -1] = torch.cumsum(scores[:, -1], dim=-1).flip(0) 
            # Last row:
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            # print('Init backward')
            # print(path_scores)
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Backward:')
            # print(path_scores)
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # First column:
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            # First row:
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            # print('Init forward')
            # print(path_scores)
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    # ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    ifread = path_scores[t, j-1] + read_cost * (j)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            # print('Forward')
            # print(path_scores)
            return path_scores

        # print('NLL scores:')
        # print(scores)
        # Forward scores:
        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        # print('Accumulated scores:')
        print(cs)
        # Find the lowest scoring path:
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1) # +1 from index to context size
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None


    def get_fb(self, net_input, target, read_cost=1):
        model = self.models[0]
        encoder_out = model.encoder(net_input['src_tokens'], net_input['src_lengths'])
        # forward in teacher-forcing
        x, attn = model.decoder.forward(net_input['prev_output_tokens'], encoder_out)
        B, Tt, Ts, V = x.size()
        scores = utils.log_softmax(x, dim=-1)  # B, Tt, Ts, V
        scores =  - scores.view(-1, V).gather(  # NLL The lowest the better
                dim=-1,
                index=target.unsqueeze(-1).expand(-1, -1, Ts).contiguous().view(-1, 1)
            ).view(B, Tt, Ts)[0]

                        
        def backward_scores(scores, read_cost):
            # Expected future cost
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # Last column:
            path_scores[:-1, -1] = torch.cumsum(scores[:, 0], dim=-1).flip(0) 
            # Last row:
            path_scores[-1] =  read_cost * torch.arange(1,  Ts+1).type_as(path_scores).flip(0) / Ts
            for t in range(Tt-1, -1, -1):
                for j in range(Ts-2, -1, -1):
                    ifwrite = path_scores[t+1, j] + scores[t,j]  # Write  (t,j) -> (t+1, j)
                    ifread = path_scores[t, j+1] + read_cost * (j+1)/Ts # Read (t,j) -> (t, j+1)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores 

        def forward_scores(scores, read_cost):
            Tt, Ts = scores.size()
            path_scores = scores.new_zeros(Tt+1, Ts)
            # First column:
            path_scores[1:, 0] = torch.cumsum(scores[:, 0], dim=-1)
            # First row:
            path_scores[0] =  read_cost * torch.arange(1, Ts+1).cumsum(dim=-1).type_as(path_scores) / Ts
            for t in range(1, Tt+1):
                for j in range(1, Ts):
                    ifwrite = path_scores[t-1, j] + scores[t-1,j]  # Write (t-1, j) -> (t, j)
                    ifread = path_scores[t, j-1] + read_cost * (j+1)/ Ts  # (t, j-1) -> (t,j)
                    path_scores[t,j] = min(
                        ifwrite, ifread
                    )
            return path_scores

        print('NLL scores:')
        print(scores)
        # Forward scores:
        fs = forward_scores(scores, read_cost)
        bs = backward_scores(scores, read_cost)
        cs = fs + bs # accumulate forward and backward
        print('Accumulated scores:')
        print(cs)
        # Find the lowest scoring path:
        t = 0 
        j = 0
        best_score = 0
        best_context = []
        while t < Tt and j < Ts-1:
            best_score += min(cs[t+1, j], cs[t, j+1])
            if cs[t+1, j] <= cs[t, j+1]: # write # Added equality in favor of writing
                best_context.append(j+1)
                t += 1
            else:  # read
                j += 1
        while len(best_context) < Tt:
            best_context.append(Ts)
        print('Best context', best_context)
        return best_context, None



    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, context_size, without_caching=False):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0],
                context_size,
                self.incremental_states,
                log_probs=True,
                without_caching=without_caching
            )

        log_probs = []
        avg_attn = None
        avg_exits = None
        avg_lengths = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn, exits, lengths = self._decode_one(
                tokens, model, encoder_out, contextt_size,
                self.incremental_states, log_probs=True,
                without_caching=without_caching,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

            if exits is not None:
                if avg_exits is not None:
                    avg_exits.add_(exits)
                else:
                    avg_exits = exits
            if lengths is not None:
                if avg_lengths is not None:
                    avg_lengths.add_(lengths)
                else:
                    avg_lengths = lengths

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        
        return avg_probs, avg_attn, avg_exits, avg_lengths

    def _decode_one(self, tokens, model, encoder_out, context_size,
                    incremental_states, log_probs,
                    without_caching=False):
        decoder_out = list(model.decoder(tokens, encoder_out,
                                         incremental_state=self.incremental_states[model],
                                         context_size=context_size,
                                         cache_decoder=not without_caching,
                                        ))

        decoder_out[0] = decoder_out[0][:, -1:, :]
        attn = None
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)

"""
transition = scores.new_ones(Tt, Ts, Ts)
for t in range(Tt):
    transition[t] = torch.triu(transition[t])
    # Normalize
transition /= transition.sum(dim=-1, keepdim=True)
transition = transition.log()
print('Full log-transition :', transition)

def _forward_alpha(scores, transition):
    Tt, Ts = scores.size()
    alpha = utils.fill_with_neg_inf(torch.empty_like(scores))  # Tt, Ts
    alpha[0] = scores[0]  - math.log(Ts)
    for i in range(1, Tt):
        alpha[i] = torch.logsumexp(alpha[i-1].unsqueeze(-1) + transition[i-1], dim=1)
        alpha[i] = alpha[i] + scores[i]
    return alpha

def _backward_beta(scores, transition):
    Tt, Ts = scores.size()
    beta = utils.fill_with_neg_inf(torch.empty_like(scores))  # Tt, Ts
    beta[-1] = 0
    for i in range(Tt-2, -1, -1):
        beta[i] = torch.logsumexp(transition[i] +  # Ts, Ts
                                  beta[i+1].unsqueeze(-1) + # Ts, 1
                                  scores[i+1].unsqueeze(-1),  # Ts, 1
                                  dim=0)

def greedy(scores):
            Tt, Ts = scores.size()
            print('Tt,Ts:', Tt, Ts)
            paths = [[j] for j in range(Ts)]   # true context is +1
            for t in range(1, Tt):
                # Expand each path with all the possible next contexts:
                expand_paths = []
                for p in paths:
                    # Above the diag
                    expand_paths.extend([p + [j] for j in range(min(Ts-1,max(p[-1], t)), Ts)])
                    # expand_paths.extend([p + [j] for j in range(p[-1], Ts)])
                paths = expand_paths
            # print('Paths:', paths)
            return paths
        
        def get_path_score(scores, p, delay_scale=1):
            Tt, Ts = scores.size()
            score = 0
            for t, j in enumerate(p):
                score += scores[t, j].data.item()
            ap = sum([zt +1 for zt in p]) / Ts  # AP
            return score/ Tt  + delay_scale *  ap / Tt  # normalize

"""

