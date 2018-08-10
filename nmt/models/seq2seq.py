"""
Sequence to Sequence with attention
Parent model.
"""
import torch.nn as nn
from .encoder import Encoder
from .cond_decoder import CondDecoder
import logging


class Seq2Seq(nn.Module):
    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size, trg_specials):
        """Initialize model."""
        nn.Module.__init__(self)
        self.logger = logging.getLogger(jobname)
        self.version = "seq2seq"
        self.params = params
        self.encoder = Encoder(params['encoder'], src_vocab_size)
        self.decoder = CondDecoder(params['decoder'], params['encoder'],
                                   trg_vocab_size, trg_specials)
        self.mapper_dropout = nn.Dropout(params['mapper']['dropout'])
        self.mapper = nn.Linear(self.encoder.size,
                                self.decoder.size)

    def init_weights(self):
        """Initialize weights."""
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.mapper.bias.data.fill_(0)

    def map(self, source):
        """ map the source code to the decoder cell size """
        # map hT^(enc) to h0^(dec)
        source['state'][0] = nn.Tanh()(self.mapper_dropout(
            self.mapper(source['state'][0])
        ))
        return source

    def forward(self, data_src, data_trg):
        source = self.encoder(data_src)
        source = self.map(source)
        logits = self.decoder(source, data_trg)
        return logits

    def sample(self, source, kwargs={}):
        """
        Sample given source with keys:
            state - ctx - emb
        """
        return self.decoder.sample(source, kwargs)

