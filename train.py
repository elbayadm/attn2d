# -*- coding: utf-8 -*-
"""
Main training loop
"""

import sys
import time
import gc
import logging
from nmt.params import parse_params, set_env


def train(params):
    jobname = params['modelname']
    _ = set_env(jobname, params['gpu_id'])
    import torch
    from torch.autograd import Variable
    from nmt.loader import ReadData
    import nmt.models.setup as ms
    from nmt.trainer import Trainer
    logger = logging.getLogger(jobname)
    # Data loading:
    logger.info('Reading data ...')
    src_loader, trg_loader = ReadData(params['data'], params['modelname'])
    src_vocab_size = src_loader.get_vocab_size()
    trg_vocab_size = trg_loader.get_vocab_size()
    trg_specials = {'EOS': trg_loader.eos,
                    'BOS': trg_loader.bos,
                    'UNK': trg_loader.unk,
                    'PAD': trg_loader.pad,
                    }
    logger.debug('trg_specials: %s', str(trg_specials))
    model = ms.build_model(jobname, params, src_vocab_size,
                           trg_vocab_size, trg_specials)
    logger.info('num. model params: %d', sum(p.data.numel()
                                             for p in model.parameters()))

    criterion = ms.define_loss(jobname, params['loss'], trg_loader)
    trainer = Trainer(jobname, params, model, criterion)

    # Recover saved epoch || idempotent jobs
    iters = trainer.load_checkpoint()
    src_loader.iterators = iters.get('src_iterators', src_loader.iterators)
    trg_loader.iterators = iters.get('trg_iterators', trg_loader.iterators)
    gc.collect()
    if trainer.lr_patient:
        trainer.update_params()
    while True:
        # update dynamic parameters: lr, alpha ...
        if not trainer.lr_patient:
            trainer.update_params()
        torch.cuda.synchronize()
        avg_loss = torch.zeros(1).cuda()
        avg_ml_loss = torch.zeros(1).cuda()
        total_ntokens = 0
        total_nseqs = 0
        start = time.time()
        for _ in range(params['optim']['num_batches']):
            data_src, order = src_loader.get_src_batch('train')
            data_trg = trg_loader.get_trg_batch('train', order)
            losses, batch_size, ntokens = trainer.step(data_src, data_trg)
            avg_loss += ntokens * losses['final']
            avg_ml_loss += ntokens * losses['ml']
            total_nseqs += batch_size
            total_ntokens += ntokens

        avg_loss /= total_ntokens
        avg_ml_loss /= total_ntokens

        trainer.backward_step(avg_loss, avg_ml_loss,
                              total_ntokens, total_nseqs,
                              start, data_src['bounds']['wrapped'])
        # sys.exit(1)
        # make evaluation on validation set, and save model
        if trainer.evaluate:
            trainer.validate(src_loader, trg_loader)
        if trainer.done:
            logger.info('Max epochs reached!')
            break


if __name__ == "__main__":
    params = parse_params()
    train(params)

