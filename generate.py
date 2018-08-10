# -*- coding: utf-8 -*-
"""
Main evaluation script
"""

import os
import os.path as osp
import logging
import time
import pickle
import json
import random
import numpy as np
import gc
from nmt.params import parse_eval_params, set_env


def generate(params):
    jobname = params['modelname'] + '_eval'
    set_env(jobname, params['gpu_id'])
    import torch
    from nmt.loader import ReadData
    import nmt.models.setup as ms
    from nmt.models.evaluate import sample_model
    # Data loading:
    logger = logging.getLogger(jobname)
    logger.info('Reading data ...')
    if params['read_length']:
        logger.warn('Max sequence length for loader: %d', params['read_length'])
        params['data']['max_src_length'] = params['read_length']
        params['data']['max_trg_length'] = params['read_length']
        evaldir = '%s/evaluations/%s_%d' % (params['modelname'], params['split'], params['read_length'])
    else:
        evaldir = '%s/evaluations/%s' % (params['modelname'], params['split'])

    if not osp.exists(evaldir):
        os.makedirs(evaldir)

    src_loader, trg_loader = ReadData(params['data'], jobname)
    src_vocab_size = src_loader.get_vocab_size()
    trg_vocab_size = trg_loader.get_vocab_size()
    trg_specials = {'EOS': trg_loader.eos,
                    'BOS': trg_loader.bos,
                    'PAD': trg_loader.pad
                    }
    # reproducibility:
    seed = params['optim']['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ms.build_model(jobname, params, src_vocab_size,
                           trg_vocab_size, trg_specials)


    # load model's weights
    if not params['last']:
        flag = '-best'
    else:
        flag = ""
    print('Loading model with flag:', flag)
    saved_state = torch.load(osp.join(params['modelname'], 'model%s.pth' % flag))
    model.load_state_dict(
        torch.load(osp.join(
            params['modelname'], 'model%s.pth' % flag)
        )
    )
    model = model.cuda()
    model.eval()
    eval_kwargs = {'split': params['split'],
                   'batch_size': params['batch_size'],
                   'beam_size': params['beam_size'],
                   # 'beam_size': 1,
                   'max_length_a': params['max_length_a'],
                   'max_length_b': params['max_length_b'],
                   'verbose': params['verbose'],
                   'block_ngram_repeat': params['block_ngram_repeat'],
                   'stepwise_penalty': params['stepwise_penalty'],
                   "normalize_length": params['norm'],
                   "lenpen": params['length_penalty'],
                   "lenpen_mode": params['length_penalty_mode'],
                   }
    params['output'] = '%s/bw%d_n%d_a%d_b%d_pl%d_%s' % (evaldir,
                                                        eval_kwargs['beam_size'],
                                                        params['batch_size'],
                                                        params['max_length_a']*10,
                                                        params['max_length_b'],
                                                        params['length_penalty']*10,
                                                        params['length_penalty_mode'])
    if params['norm']:
        params['output'] += "_norm"

    if params['last']:
        params['output'] = params['output'] + '_last'
    if params['edunov']:
        params['output'] += "_edunov"

    print('Evaluation settings:', eval_kwargs)
    preds, bleu = sample_model(jobname,
                               model, src_loader, trg_loader,
                               eval_kwargs)
    logger.warn('Bleu (split=%s, beam=%d) : %.3f' % (eval_kwargs['split'],
                                                     eval_kwargs['beam_size'],
                                                     bleu))
    with open(params['output'] + '.res', 'w') as f:
        f.write(str(bleu))
    json.dump(preds, open(params['output'] +
                          '.json', 'w',
                          encoding='utf8'),
              ensure_ascii=False)


if __name__ == "__main__":
    params = parse_eval_params()
    generate(params)

