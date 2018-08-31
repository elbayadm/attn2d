import logging
import json
import pickle
import h5py
import numpy as np
import torch
from .split_dataloader import SplitLoader


class DataPair(object):
    """
    Wrapper around the two languages' loaders
    """
    def __init__(self, params, split, job_name, verbose=False):
        self.split = split
        self.logger = logging.getLogger(job_name)
        self.max_batch_size = params['max_batch_size']
        self.max_tokens = params["max_tokens"]
        ddir = params['dir']
        src = params['src']
        trg = params['trg']
        dataparams = {'h5': "%s/%s_%s.%s" % (ddir, src, split, "h5"),
                      'infos': "%s/%s.%s" % (ddir, src, "infos"),
                      'batch_size': params['batch_size'],
                      'max_length': params['max_src_length'],
                      }

        self.src = SplitLoader(dataparams, job_name=job_name, verbose=verbose)

        dataparams = {'h5': "%s/%s_%s.%s" % (ddir, trg, split, "h5"),
                      'infos': "%s/%s.%s" % (ddir, trg, "infos"),
                      'batch_size': params['batch_size'],
                      'max_length': params['max_trg_length'],
                      }

        self.trg = SplitLoader(dataparams, job_name=job_name, verbose=verbose)

    def get_batch(self):
        it = self.trg.iterators
        assert it == self.src.iterators
        # pick the batch_size:
        trg_sizes = self.trg.h5_file["lengths"][it: it + self.max_batch_size]
        src_sizes = self.src.h5_file["lengths"][it: it + self.max_batch_size]
        # self.logger.warn('src: %s', str(src_sizes))
        # self.logger.warn('trg: %s', str(trg_sizes))
        # self.logger.warn('dot: %s', str(trg_sizes * src_sizes))
        cumul = np.cumsum(src_sizes * trg_sizes)
        # self.logger.warn('cumul: %s', str(cumul))
        batch_size = np.argmax(cumul > self.max_tokens)
        # self.logger.warn('cumul at (%d) is %d', batch_size, cumul[batch_size])
        if batch_size < 1:
            # self.logger.warn('cumul: %s', str(cumul))
            # self.logger.warn('smaller than 1!!, using the max batch')
            if cumul[-1] < self.max_tokens:
                batch_size = self.max_batch_size
            else:
                batch_size = 2
            # diff = np.ediff1d(sizes[:batch_size])
        # self.logger.warn('trg sizes type: %s', str(trg_sizes.dtype))
        diff = trg_sizes[:batch_size] - trg_sizes[0]
        # self.logger.warn('diff: %s', str(diff))
        if sum(diff):
            batch_size = np.argmax(diff >= 1)
            # self.logger.warn('diff: %s', str(diff))
            # self.logger.warn('trg sizes: %s', str(trg_sizes[:batch_size]))
            # self.logger.warn('picked batch size: %d', batch_size)

        data_trg, order = self.trg.get_trg_batch(batch_size)
        data_src, _ = self.src.get_src_batch(batch_size, order)
        ntokens = (data_trg['lengths'] * data_src['lengths']).sum().data.item()
        # print("batch:", data_src['labels'].size(0),
              # 'SRC:', data_src['lengths'].sum().data.item(),
              # 'TRG:', data_trg['lengths'].sum().data.item(),
              # "Cartesian:", ntokens)
        sample = {"src": data_src,
                  "trg": data_trg,
                  "ntokens": ntokens}
        return sample

    def reset_iterator(self, split=None):
        self.src.reset_iterator()
        self.trg.reset_iterator()





