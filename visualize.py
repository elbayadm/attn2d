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
    jobname = params['modelname'] + '_track'
    set_env(jobname, params['gpu_id'])
    import torch
    from nmt.loader import ReadData
    import nmt.models.setup as ms
    from nmt.models.evaluate import track_model
    # Data loading:
    logger = logging.getLogger(jobname)
    logger.info('Reading data ...')
    src_loader, trg_loader = ReadData(params['data'], jobname)
    src_vocab_size = src_loader.get_vocab_size()
    trg_vocab_size = trg_loader.get_vocab_size()
    trg_specials = {'EOS': trg_loader.eos,
                    'BOS': trg_loader.bos,
                    'PAD': trg_loader.pad
                    }
    evaldir = '%s/evaluations/%s' % (params['modelname'], params['split'])
    if not osp.exists(evaldir):
        os.makedirs(evaldir)

    # reproducibility:
    seed = params['optim']['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ms.build_model('eval',
                           params, src_vocab_size,
                           trg_vocab_size, trg_specials)


    # load model's weights
    if not params['last']:
        flag = '-best'
    else:
        flag = ""
    print('Loading model with flag:', flag)
    model.load_state_dict(
        torch.load(osp.join(params['modelname'],
                            'model%s.pth' % flag)
                   )
    )
    model = model.cuda()
    model.eval()
    eval_kwargs = {'split': params['split'],
                   'batch_size': params['batch_size'],
                   'beam_size': params['beam_size'],
                   'max_length_a': params['max_length_a'],
                   'max_length_b': params['max_length_b'],
                   'verbose': 1,
                   "normalize_length": params['norm'],
                   "max_samples": params['max_samples'],
                   "offset": params["offset"]}

    params['output'] = '%s/activat_o%d' % (evaldir, params['offset'])

    # params['output'] = '%s/bw%d_n%d_a%d_b%d' % (evaldir, eval_kwargs['beam_size'],
                                                # params['batch_size'],
                                                # params['max_length_a']*10,
                                                # params['max_length_b'])
    if params['norm']:
        params['output'] += "_norm"

    if params['last']:
        params['output'] = params['output'] + '_last'

    outs = track_model(jobname,
                       model, src_loader, trg_loader,
                       eval_kwargs)
    logger.warn('Bleu (split=%s, beam=%d) : %.3f' % (eval_kwargs['split'],
                                                     eval_kwargs['beam_size'],
                                                     outs['bleu']))
    pickle.dump(outs, open(params['output'] + '.attn', 'wb'))


if __name__ == "__main__":
    params = parse_eval_params()
    generate(params)

