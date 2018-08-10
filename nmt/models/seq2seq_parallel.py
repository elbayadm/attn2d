"""
Sequence to Sequence with attention
Parent model.
"""
import torch.nn as nn
from .seq2seq import Seq2Seq
from .encoder_parallel import Encoder_Parallel
from .cond_decoder import CondDecoder
import logging


class Seq2Seq_Parallel(nn.Module):
    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size, trg_specials):
        """Initialize model."""
        # model = Seq2Seq(jobname, params, src_vocab_size, trg_vocab_size, trg_specials)
        nn.Module.__init__(self)
        self.logger = logging.getLogger(jobname)
        self.version = "seq2seq"
        self.params = params
        self.encoder = Encoder_Parallel(params['encoder'], src_vocab_size)
        # self.encoder = nn.DataParallel(encoder)
        decoder = CondDecoder(params['decoder'], params['encoder'],
                              trg_vocab_size, trg_specials)
        # self.decoder = nn.DataParallel(decoder)
        self.decoder = decoder
        mapper = nn.Linear(self.encoder.size,
                           self.decoder.size)
                           # self.decoder.module.size)

        self.mapper = nn.DataParallel(mapper)
        self.mapper_dropout = nn.Dropout(params['mapper']['dropout'])


    def init_weights(self):
        """Initialize weights."""
        self.encoder.init_weights()
        # self.decoder.module.init_weights()
        self.decoder.init_weights()
        self.mapper.module.bias.data.fill_(0)

    def map(self, source):
        """ map the source code to the decoder cell size """
        # map hT^(enc) to h0^(dec)
        source['state'][0] = nn.Tanh()(self.mapper_dropout(
            self.mapper(source['state'][0])
        ))
        return source

    def forward(self, data_src, data_trg):
        source = self.encoder(data_src)
        # self.logger.info('Source: %s', source['ctx'].size())
        source = self.map(source)
        # self.logger.info('Mapped source: %s', source['ctx'].size())
        # self.logger.info('Feeding the decoder: %s', data_trg['labels'].size())
        logits = self.decoder(source, data_trg)
        return logits

    def sample(self, source, kwargs={}):
        """
        Sample given source with keys:
            state - ctx - emb
        """
        # return self.decoder.module.sample(source, kwargs)
        return self.decoder.sample(source, kwargs)


