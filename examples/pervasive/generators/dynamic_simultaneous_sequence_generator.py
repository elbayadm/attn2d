import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from torch import Tensor


READ=0
WRITE=1


class DynamicPervasiveSequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        match_source_len=False,
        eos=None,
        shift=1,
        write_threshold=0.5,
    ):
        """
        Greedy one-sentence SequenceGenerator following read/write decisions
        shift : initial context
        write_threshold : threshold to decide between reading and writing

        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.shift = shift
        self.write_threshold = write_threshold

        self.match_source_len = match_source_len
        self.model.eval()


    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        self.model.reset_incremental_state()
        return self._generate(sample, prefix_tokens, bos_token)

    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        self.model.reset_incremental_state()
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):

        encoder_input: Dict[str, Tensor] = {}
        for k, v in sample["net_input"].items():
            if k != "prev_output_tokens":
                encoder_input[k] = v

        src_tokens = encoder_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(
            src_tokens=encoder_input["src_tokens"],
            src_lengths=encoder_input["src_lengths"],
        )

        tokens = [self.eos]
        scores = []

        finished = False
        step = 0
        ctx = self.shift
        max_read_len = src_len + 1
        context = []
        while not finished:
            action = READ
            while action == READ:
                if ctx >= max_read_len:
                    action = WRITE
                else:
                    action = self.model.get_action(
                            src_tokens.new_tensor(tokens).long().unsqueeze(0),
                            encoder_outs,
                            ctx,  # Read ctx tokens
                            self.write_threshold,
                        )
                    if action == READ:
                        ctx += 1

            # Forward step 
            lprobs, avg_attn_scores = self.model.forward_decoder(
                src_tokens.new_tensor(tokens).long().unsqueeze(0),
                encoder_outs, 
                ctx,  # context size
            )
            lprobs[lprobs != lprobs] = -math.inf
            # Forbid a few tokens:
            lprobs[:, self.pad] = -math.inf 
            lprobs[:, self.unk] = -math.inf
            context.append(ctx)

            # Pick a candidate with a score.
            cand_token_score, cand_token_index = lprobs.max(dim=-1)
            tokens.append(cand_token_index.item())
            scores.append(cand_token_score.item())
            if cand_token_index == self.eos or step > max_len:
                finished = True
            # increment the step:
            step += 1
            
        # could be better but this will do.
        return [[{
                'tokens': torch.Tensor(tokens[1:]), # Remove BOS
                'score': sum(scores)/len(scores),
                'positional_scores': torch.Tensor(scores),
                'alignment' : None,
                'context': [min(c,src_len) for c in context] # Do not account for EOS
            }]]


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]]

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.models_size)
            ],
        )
        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def reset_incremental_state(self):
        if self.has_incremental_states():
            self.incremental_states = torch.jit.annotate(
                List[Dict[str, Dict[str, Optional[Tensor]]]],
                [
                    torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                    for i in range(self.models_size)
                ],
            )
        return

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, src_tokens, src_lengths):
        if not self.has_encoder():
            return None
        return [
            model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            for model in self.models
        ]

    @torch.jit.export
    def get_action(self, tokens, encoder_outs, context_size, threshold=0.5):
        # Should I read or should I write:
        write_proba = self.models[0].decide(tokens, encoder_outs[0], context_size)
        print('P(w)', write_proba.item(), 'th:', threshold)
        if write_proba > threshold:
            return WRITE
        return READ

    @torch.jit.export
    def forward_decoder(
        self, tokens, encoder_outs: List[EncoderOut], ctx=None, temperature=1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=self.incremental_states[i],
                    context_size=ctx,
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn
