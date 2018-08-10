import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AllamanisConvAttention(nn.Module):
    """
    Convolutional attention
    @inproceedings{allamanis2016convolutional,
          title={A Convolutional Attention Network for
                 Extreme Summarization of Source Code},
          author={Allamanis, Miltiadis and Peng, Hao and Sutton, Charles},
          booktitle={International Conference on Machine Learning (ICML)},
          year={2016}
      }
    """

    def __init__(self, params, enc_params):
        super(AllamanisConvAttention, self).__init__()
        src_emb_dim = enc_params['input_dim']
        dims = params['attention_channels'].split(',')
        dim1, dim2 = [int(d) for d in dims]
        print('Out channels dims:', dim1, dim2)
        widths = params['attention_windows'].split(',')
        w1, w2, w3 = [int(w) for w in widths]
        print('Moving windows sizes:', w1, w2, w3)
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        # padding to maintaing the same length
        self.conv1 = nn.Conv1d(src_emb_dim, dim1, w1, padding=(w1-1)//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, trg_dim, w2, padding=(w2-1)//2)
        self.conv3 = nn.Conv1d(trg_dim, 1, w3, padding=(w3-1)//2)
        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_emb_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def score(self, input, context, src_emb):
        """
        input: batch x trg_dim
        context & src_emb : batch x Tx x src_dim (resp. src_emb_dim)
        return the alphas for comuting the weighted context
        """
        src_emb = src_emb.transpose(1, 2)
        L1 = self.relu(self.conv1(src_emb))
        L2 = self.conv2(L1)
        # columnwise multiplication
        L2 = L2 * input.unsqueeze(2).repeat(1, 1, L2.size(2))
        # L2 normalization:
        if self.normalize:
            norm = L2.norm(p=2, dim=1, keepdim=True)  # check if 2 is the right dim
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                print('Zero norm!!')
        attn = self.conv3(L2)
        attn_sm = self.sm(attn)
        return attn_sm

    def forward(self, input, context, src_emb):
        """
        Score the context (resp src embedding)
        and return a new context as a combination of either
        the source embeddings or the hidden source codes
        """
        attn_sm = self.score(input, context, src_emb)
        weighted_context = torch.bmm(attn_sm, src_emb).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn_sm


class AllamanisConvAttentionBis(AllamanisConvAttention):
    """
    Similar to AllamanisConvAttention with the only difference at computing
    the weighted context which takes the encoder's hidden states
    instead of the source word embeddings
    """

    def __init__(self, params, enc_params):
        super(AllamanisConvAttentionBis, self).__init__(params)
        trg_dim = params['cell_dim']
        src_dim = enc_params['cell_dim']
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)


    def forward(self, input, context, src_emb):
        attn_sm = self.score(input, context, src_emb)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context.transpose(1, 2),
                                     attn_reshape).squeeze(2)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn_sm


class ConvAttentionHid(nn.Module):
    """
    Convolutional attention
    All around similar to Allamanis attention while never
    using the source word embeddings
    """

    def __init__(self, params, enc_params):
        super(ConvAttentionHid, self).__init__()
        src_dim = enc_params['cell_dim']
        dims = params['attention_channels'].split(',')
        dim1, dim2 = [int(d) for d in dims]
        print('Out channels dims:', dim1, dim2)
        widths = params['attention_windows'].split(',')
        w1, w2, w3 = [int(w) for w in widths]
        print('Moving windows sizes:', w1, w2, w3)
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        # padding to maintaing the same length
        self.conv1 = nn.Conv1d(src_dim, dim1, w1, padding=(w1-1)//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, trg_dim, w2, padding=(w2-1)//2)
        self.conv3 = nn.Conv1d(trg_dim, 1, w3, padding=(w3-1)//2)
        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input, context, src_emb):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        context = context.transpose(1, 2)
        L1 = self.relu(self.conv1(context))
        L2 = self.conv2(L1)
        # columnwise dot product
        L2 = L2 * input.unsqueeze(2).repeat(1, 1, L2.size(2))
        # L2 normalization:
        if self.normalize:
            norm = L2.norm(p=2, dim=2, keepdim=True)
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                print('Zero norm!!')
        attn = self.conv3(L2)
        attn_sm = self.sm(attn)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context,
                                     attn_reshape).squeeze(2)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ConvAttentionHidCat(nn.Module):
    """
    Convolutional attention
    Use the encoder hidden states all around, Jakob's idea
    """

    def __init__(self, params, enc_params):
        super(ConvAttentionHidCat, self).__init__()
        src_dim = enc_params['cell_dim']
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        widths = params['attention_windows'].split(',')
        self.num_conv_layers = len(widths)
        dims = params['attention_channels'].split(',')
        assert len(dims) == self.num_conv_layers - 1
        if self.num_conv_layers == 3:
            w1, w2, w3 = [int(w) for w in widths]
            print('Moving windows sizes:', w1, w2, w3)
            dim1, dim2 = [int(d) for d in dims]
            print('Out channels dims:', dim1, dim2)
        elif self.num_conv_layers == 4:
            w1, w2, w3, w4 = [int(w) for w in widths]
            print('Moving windows sizes:', w1, w2, w3, w4)
            dim1, dim2, dim3 = [int(d) for d in dims]
            print('Out channels dims:', dim1, dim2, dim3)
        else:
            raise ValueError('Number of layers is either 3 or 4, still working on a general form')
        # padding to maintaing the same length
        self.conv1 = nn.Conv1d(src_dim + trg_dim, dim1, w1, padding=(w1-1)//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, dim2, w2, padding=(w2-1)//2)
        if self.num_conv_layers == 3:
            self.conv3 = nn.Conv1d(dim2, 1, w3, padding=(w3-1)//2)
        elif self.num_conv_layers == 4:
            self.conv3 = nn.Conv1d(dim2, dim3, w3, padding=(w3-1)//2)
            self.conv4 = nn.Conv1d(dim3, 1, w4, padding=(w4-1)//2)

        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input, context, src_emb):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        context = context.transpose(1, 2)
        input_cat = torch.cat((context,
                               input.unsqueeze(2).repeat(1,
                                                         1,
                                                         context.size(2))),
                              1)
        L1 = self.relu(self.conv1(input_cat))
        L2 = self.conv2(L1)
        # L2 normalization:
        if self.normalize:
            norm = L2.norm(p=2, dim=2, keepdim=True)
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                print('Zero norm!!')
        # print('L2 normalized:', L2.size())
        if self.num_conv_layers == 3:
            attn = self.conv3(L2)
        else:
            attn = self.conv4(self.conv3(L2))
        attn_sm = self.sm(attn)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context,
                                     attn_reshape).squeeze(2)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LocalDotAttention(nn.Module):
    """
    Soft Dot/ local-predictive attention

    Ref: http://www.aclweb.org/anthology/D15-1166
    Effective approaches to attention based NMT (Luong et al. EMNLP 15)
    """

    def __init__(self, params):
        super(LocalDotAttention, self).__init__()
        dim = params['cell_dim']
        dropout = params['attention_dropout']
        self.window = 4  # D
        self.sigma = self.window / 2
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.linear_predict_1 = nn.Linear(dim, dim//2, bias=False)
        self.linear_predict_2 = nn.Linear(dim//2, 1, bias=False)

        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, context, src_emb=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        Tx = context.size(1)
        # predict the alignement position:
        pt = self.tanh(self.linear_predict_1(input))
        print('pt:', pt.size())
        pt = self.linear_predict_2(pt)
        print('pt size:', pt.size())
        pt = Tx * self.sigmoid(pt)
        bl, bh = (pt-self.window).int(), (pt+self.window).int()
        indices = torch.cat([torch.arange(i.item(), j.item()).unsqueeze(0)
                             for i, j in zip(bl, bh)],
                            dim=0).long().cuda()
        print('indices:', indices.size())
        # Get attention
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        print('type(context):', type(context))
        context_window = context.gather(0, indices)
        print('context window:', context_window.size())
        attn = torch.bmm(context_window, target).squeeze(2)  # batch x sourceL
        attn = self.sm(self.dropout(attn))
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weighted_context = torch.bmm(attn3, context_window).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Effective approaches to attention based NMT (Luong et al. EMNLP 15)
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, params):
        super(SoftDotAttention, self).__init__()
        dim = params['cell_dim']
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(params['attention_dropout'])

    def forward(self, input, context, src_emb=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(self.dropout(attn))
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttention(nn.Module):
    """
    A long short-term memory (LSTM) cell with attention.
    Use SoftDotAttention
    """

    def __init__(self, params, enc_params):
        super(LSTMAttention, self).__init__()
        # Params:
        self.mode = params['attention_mode']
        self.input_size = params['input_dim']
        self.hidden_size = params['cell_dim']
        self.input_weights = nn.Linear(self.input_size,
                                       4 * self.hidden_size)
        self.hidden_weights = nn.Linear(self.hidden_size,
                                        4 * self.hidden_size)

        if self.mode == "dot":
            self.attention_layer = SoftDotAttention(params)
        elif self.mode == "local-dot":
            self.attention_layer = LocalDotAttention(params)
        elif self.mode == "allamanis":  # conv
            self.attention_layer = AllamanisConvAttention(params, enc_params)
        elif self.mode == "allamanis-v2":  # conv2
            self.attention_layer = AllamanisConvAttentionBis(params, enc_params)
        elif self.mode == "conv-hid":   # conv3
            self.attention_layer = ConvAttentionHid(params, enc_params)
        elif self.mode == "conv-hid-cat":  # conv4
            self.attention_layer = ConvAttentionHidCat(params, enc_params)
        else:
            raise ValueError('Unkown attention mode %s' % self.mode)


    def forward(self, input, hidden, ctx, src_emb):
        """Propogate input through the network."""
        # print('input:', input.size())
        # print('hidden:', hidden[0].size(), hidden[1].size())
        # print('ctx:', ctx.size())

        def recurrence(input, hidden):
            """Recurrence helper."""
            # print('hidden', hidden[0].size(), hidden[1].size())
            hx, cx = hidden  # n_b x hidden_dim #FIXME Assuming LSTM
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, attn = self.attention_layer(hy, ctx, src_emb)
            return (h_tilde, cy), attn

        input = input.transpose(0, 1)
        output = []
        attention = []
        steps = list(range(input.size(0)))
        for i in steps:
            hidden, attn = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                h = hidden[0]
            else:
                h = hidden
            output.append(h)
            attention.append(attn)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        attention = torch.cat(attention, 0)
        return output, hidden, attention


class LSTMAttentionV2(LSTMAttention):
    """
    A long short-term memory (LSTM) cell with attention.
    Use SoftDotAttention
    """

    def __init__(self, params, enc_params):
        super(LSTMAttentionV2, self).__init__(params, enc_params)

    def forward(self, input, hidden, ctx, src_emb):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim #FIXME Assuming LSTM
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, _ = self.attention_layer(hy, ctx, src_emb)
            return h_tilde, (hy, cy)

        input = input.transpose(0, 1)
        output = []
        steps = list(range(input.size(0)))
        for i in steps:
            htilde, hidden = recurrence(input[i], hidden)
            output.append(htilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden

class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)

    def forward(self, input, hidden, ctx, src_emb):
        if hidden[0].size(0) != 1:
            # print('unsqueezing the state')
            hidden = [h.unsqueeze(0) for h in hidden]
        # print('input:', input.size(), 'hidden:', hidden[0].size())
        output, hdec = super(LSTM, self).forward(input, hidden)
        # print('out & hid:', output.size(), hdec[0].size())
        return output, hdec

