import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .beam_search import Beam
from .beam_onmt import Beam as Beam_ONMT
from .lstm import LSTMAttention, LSTMAttentionV2, LSTM


class CondDecoder(nn.Module):
    def __init__(self, params, enc_params, vocab_size, special_tokens):
        nn.Module.__init__(self)
        self.input_dim = params['input_dim']
        self.size = params['cell_dim']
        self.nlayers = 1
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.eos_token = special_tokens['EOS']
        self.bos_token = special_tokens['BOS']
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.input_dim,
            self.pad_token,
            scale_grad_by_freq=bool(params['scale_grad_by_freq'])
        )
        self.input_dropout = nn.Dropout(params['input_dropout'])
        # print('Setting attention!', params['attention_mode'])
        if params["attention_mode"] == "none":
            self.cell = LSTM(self.input_dim,
                             self.size,
                             self.nlayers,
                             batch_first=True,
                             )
        else:
            if params['state_update'] == 1:
                self.cell = LSTMAttention(params, enc_params)
            elif params['state_update'] == 2:
                self.cell = LSTMAttentionV2(params, enc_params)

        self.prediction_dropout = nn.Dropout(params['prediction_dropout'])
        self.prediction = nn.Linear(self.size,
                                    self.vocab_size)

    def init_weights(self):
        """Initialize weights."""
        initdev = 0.01
        self.embedding.weight.data.normal_(0.0, initdev)
        self.prediction.bias.data.fill_(0)

    def forward_(self, source, data):
        labels = data['labels']
        emb = self.input_dropout(self.embedding(labels))
        h, (_, _), _ = self.cell(
            emb,
            source['state'],
            source['ctx'],
            source['emb']
        )

        # print('decoder hidden stats:', h.size())
        return h

    def forward(self, source, data):
        labels = data['labels']
        emb = self.input_dropout(self.embedding(labels))
        h, (_, _), _ = self.cell(
            emb,
            source['state'],
            source['ctx'],
            source['emb']
        )

        h_reshape = h.contiguous().view(
            h.size()[0] * h.size()[1],
            h.size()[2]
        )
        logits = F.log_softmax(
            self.prediction(
                self.prediction_dropout(
                    h_reshape)),
            dim=1)
        logits = logits.view(
            h.size(0),
            h.size(1),
            logits.size(1)
        )
        return logits

    def sample_beam_new(self, source, scorer=None, kwargs={}):
        beam_size = kwargs.get('beam_size', 3)
        state = source['state']
        source["emb"] = source["emb"].repeat(beam_size, 1, 1)
        source["ctx"] = source["ctx"].repeat(beam_size, 1, 1)
        batch_size = state[0].size(0)
        dec_states = [state[0].repeat(beam_size, 1),
                      state[1].repeat(beam_size, 1)]

        # remap special tokens:
        beam_args = {}
        for k in ['EOS', 'BOS', 'PAD']:
            beam_args[k.lower()] = kwargs[k]

        beam = [Beam(beam_size, kwargs) for k in range(batch_size)]
        beam = [Beam_ONMT(beam_size, **beam_args,
                          cuda=True,
                          global_scorer=scorer,
                          block_ngram_repeat=kwargs['block_ngram_repeat'],
                          exclusion_tokens=set([self.eos_token,
                                                self.bos_token]),
                          stepwise_penalty=kwargs['stepwise_penalty'])
                for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        max_length = kwargs.get('max_length', 50)
        for t in range(max_length):
            input = torch.stack([b.get_current_state()
                                 for b in beam if not b.done()]
                                ).t().contiguous().view(1, -1)

            emb = self.embedding(input.transpose(1, 0))
            h, dec_states, attn = self.cell(emb,
                                            dec_states,
                                            source['ctx'],
                                            source['emb'])
            h_reshape = h.contiguous().view(
                h.size()[0] * h.size()[1],
                h.size()[2]
            )
            out = F.log_softmax(
                self.prediction(h_reshape),
                dim=1)

            word_lk = out.view(beam_size,
                               remaining_sents,
                               -1).transpose(0, 1).contiguous()
            active = []
            for b in range(batch_size):
                if beam[b].done():
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], attn):  #FIXME
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam * sent x dim
                    dec_size = dec_state.size()
                    sent_states = dec_state.view(
                        beam_size, remaining_sents, dec_size[-1]
                    )[:, idx, :]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            0,
                            beam[b].get_current_origin()
                        )
                    )
            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.contiguous().view(
                    -1, remaining_sents,
                    # self.model.decoder.hidden_size
                    t.size(-1)
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            source['ctx'] = update_active(source['ctx'].t()).t()
            source['emb'] = update_active(source['emb'].t()).t()

            remaining_sents = len(active)

        # Wrap up
        allHyp, allScores = [], []
        n_best = 1
        for b in range(batch_size):
            scores, ks = beam[b].sort_finished(n_best)
            # print('scores:', scores)
            # print('ks:', ks)
            allScores += [scores[:n_best]]
            # hyps = list(zip(*[beam[b].get_hyp(k) for k in ks[:n_best]]))
            hyps, _ = beam[b].get_hyp(*ks[0])
            allHyp += [hyps]
        return allHyp, allScores

    def sample_beam(self, source, scorer=None, kwargs={}):
        beam_size = kwargs.get('beam_size', 3)
        state = source['state']
        source["emb"] = source["emb"].repeat(beam_size, 1, 1)
        source["ctx"] = source["ctx"].repeat(beam_size, 1, 1)
        batch_size = state[0].size(0)
        dec_states = [state[0].repeat(beam_size, 1),
                      state[1].repeat(beam_size, 1)]

        beam = [Beam(beam_size, kwargs) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        max_length = kwargs.get('max_length', 50)
        for t in range(max_length):
            input = torch.stack([b.get_current_state()
                                 for b in beam if not b.done]
                                ).t().contiguous().view(1, -1)

            emb = self.embedding(input.transpose(1, 0))
            h, dec_states, _ = self.cell(emb,
                                         dec_states,
                                         source['ctx'],
                                         source['emb'])
            h_reshape = h.contiguous().view(
                h.size()[0] * h.size()[1],
                h.size()[2]
            )
            out = F.log_softmax(
                self.prediction(h_reshape),
                dim=1)

            word_lk = out.view(beam_size,
                               remaining_sents,
                               -1).transpose(0, 1).contiguous()
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], t):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam * sent x dim
                    dec_size = dec_state.size()
                    sent_states = dec_state.view(
                        beam_size, remaining_sents, dec_size[-1]
                    )[:, idx, :]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            0,
                            beam[b].get_current_origin()
                        )
                    )
            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.contiguous().view(
                    -1, remaining_sents,
                    # self.model.decoder.hidden_size
                    t.size(-1)
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            source['ctx'] = update_active(source['ctx'].t()).t()
            source['emb'] = update_active(source['emb'].t()).t()

            remaining_sents = len(active)

        # Wrap up
        allHyp, allScores = [], []
        n_best = 1
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            # hyps = list(zip(*[beam[b].get_hyp(k) for k in ks[:n_best]]))
            hyps = beam[b].get_hyp(ks[0])
            allHyp += [hyps]
        return allHyp, allScores

    def sample(self, source, scorer=None, kwargs={}):
        beam_size = kwargs.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(source, scorer, kwargs)
            # return self.sample_beam_new(source, scorer, kwargs)

        state = source['state']
        batch_size = state[0].size(0)
        max_length = kwargs.get('max_length', 50)
        seq = []
        scores = None
        for t in range(max_length):
            if t == 0:
                input = torch.LongTensor([[self.bos_token]
                                          for i in range(batch_size)
                                          ]).cuda()
            emb = self.embedding(input)
            h, state, _ = self.cell(
                emb,
                state,
                source['ctx'],
                source['emb']
            )
            h_reshape = h.contiguous().view(
                h.size()[0] * h.size()[1],
                h.size()[2]
            )
            logits = F.log_softmax(
                self.prediction(h_reshape),
                dim=1)[:, 1:]  # remove the proba of padding
            np_logits = logits.data.cpu().numpy()
            decoder_argmax = 1 + np_logits.argmax(axis=-1)
            if t:
                scores += np_logits[:, decoder_argmax - 1]
            else:
                scores = np_logits[:, decoder_argmax - 1]
            next_preds = torch.from_numpy(decoder_argmax).view(-1, 1).cuda()
            seq.append(next_preds)
            input = next_preds
            if t >= 2:
                # stop when all finished
                unfinished = torch.add(
                        torch.mul((input ==
                                   self.eos_token
                                   ).type_as(logits), -1), 1)
                if unfinished.sum().data[0] == 0:
                    break

        seq = torch.cat(seq, 1).data.cpu().numpy()
        return seq, scores
