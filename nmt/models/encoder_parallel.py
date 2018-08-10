import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

"""
An (bi-)LSTM/GRU encoder
"""
class Cell_Parallell(nn.Module):
    # ... __init__, other methods, etc.

    # padding_input is of shape [B x T x *] (batch_first mode) and contains
    # the sequences sorted by lengths
    # B is the batch size
    # T is max sequence length
    def __init__(self, cell, cell_type):
        nn.Module.__init__(self)
        self.cell = cell
        self.cell_type = cell_type

    def forward(self, padded_input, input_lengths):
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input,
                                            input_lengths,
                                            batch_first=True)
        packed_output, state = self.cell(packed_input)
        # Permute to gather correctly
        h, c = state
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        # print('state with CellP:', state[0].size(), state[1].size(), type(state))

        # print('packed output:', packed_output)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        # print('ctx with CellP:', output.size())

        return output, h, c


class Encoder_Parallel(nn.Module):
    def __init__(self, params, vocab_size):
        nn.Module.__init__(self)
        # input
        self.input_dim = params['input_dim']
        self.vocab_size = vocab_size
        self.pad_token = 0
        # cell
        self.bidirectional = params['bidirectional']
        self.nd = 2 if self.bidirectional else 1
        self.cell_type = params['cell_type'].upper()
        self.nlayers = params['num_layers']
        self.size = params['cell_dim']
        # if bidirectional split
        self.hidden_dim = self.size // self.nd

        # layers
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.input_dim,
            self.pad_token,
            scale_grad_by_freq=bool(params['scale_grad_by_freq'])
        )

        self.input_dropout = nn.Dropout(params['input_dropout'])
        if params['cell_dropout'] and self.nlayers == 1:
            # dropout effective only if nlyaers > 1
            params['cell_dropout'] = 0
        cell = getattr(nn,
                       self.cell_type)(
                           self.input_dim,
                           self.hidden_dim,
                           self.nlayers,
                           bidirectional=self.bidirectional,
                           batch_first=True,
                           dropout=params['cell_dropout']
                       )

        self.cell = nn.DataParallel(Cell_Parallell(cell, self.cell_type))

    def init_weights(self):
        """Initialize weights."""
        initdev = 0.01
        self.embedding.weight.data.normal_(0.0, initdev)
        self.embedding.weight.data.normal_(0.0, initdev)
        # FIXME add the option of initializing with loaded weights and freeze

    def init_state(self, batch_size):
        """Get cell states and hidden states."""
        h0 = torch.zeros(
            self.nlayers * self.nd,
            batch_size,
            self.hidden_dim
        )
        if self.cell_type == 'GRU':
            return h0.cuda()

        c0 = torch.zeros(
            self.nlayers * self.nd,
            batch_size,
            self.hidden_dim
        )
        return h0.cuda(), c0.cuda()


    def forward(self, data):
        labels = data['labels']
        lengths = data['lengths']
        batch_size = labels.size(0)
        emb = self.input_dropout(self.embedding(labels))
        _emb = emb  # to pass in case needed for attenion scores
        # pack_emb = pack_padded_sequence(emb,
                                        # lengths,
                                        # batch_first=True)

        # state = self.init_state(batch_size)
        ctx, h, c = self.cell(emb, lengths)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        state = (h, c)
        # print('encoder forward ctx:', ctx.size())
        # print('encoder forward state:', state[0].size(), state[1].size())

        # unpack
        # ctx, _ = pad_packed_sequence(ctx,
                                     # batch_first=True)
        if self.bidirectional:
            if self.cell_type == "LSTM":
                h_t = torch.cat((state[0][-1],
                                 state[0][-2]), 1)
                c_t = torch.cat((state[1][-1],
                                 state[1][-2]), 1)
                final_state = [h_t, c_t]

            elif self.cell_type == "GRU":
                h_t = torch.cat((state[-1],
                                 state[-2]), 1)
                final_state = [h_t]

        else:
            if self.cell_type == "LSTM":
                h_t = state[0][-1]
                c_t = state[1][-1]
                final_state = [h_t, c_t]

            elif self.cell_type == "GRU":
                h_t = state[0][-1]
                final_state = [h_t]
        return {"emb": _emb, "ctx": ctx, "state": final_state}

