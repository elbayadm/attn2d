import logging
import json
import pickle
import h5py
import numpy as np
import torch

class textDataLoader(object):
    """
    Text data iterator class
    """
    def __init__(self, params, job_name):
        self.logger = logging.getLogger(job_name)
        infos = pickle.load(open(params['infos'], 'rb'))
        self.ix_to_word = infos['itow']
        self.vocab_size = len(self.ix_to_word)
        self.ref = params["h5"]
        self.logger.warn('vocab size is %d ' % self.vocab_size)
        # open the hdf5 file
        self.logger.warn('DataLoader loading h5 file: %s' % params['h5'])
        self.h5_file = h5py.File(params['h5'])
        self.logger.warn('Training set length: %d' % self.h5_file['labels_train'].shape[0])
        self.logger.warn('Validation set length: %d' % self.h5_file['labels_val'].shape[0])
        self.logger.warn('Test set length: %d' % self.h5_file['labels_test'].shape[0])
        # load in the sequence data
        self.batch_size = params['batch_size']
        seq_size = self.h5_file['labels_test'].shape
        self.logger.warn('saved sequence length in data is %d' % seq_size[1])
        self.seq_length = params['max_length']
        self.logger.warn('max sequence length in data is %d' % self.seq_length)
        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        word_to_ix = {w: ix for ix, w in self.ix_to_word.items()}
        self.pad = word_to_ix['<PAD>']
        self.unk = word_to_ix['<UNK>']
        try:
            self.eos = word_to_ix['<EOS>']
            self.bos = word_to_ix['<BOS>']
        except:
            self.eos = self.pad
            self.bos = self.pad
        self.logger.debug('Special tokens: PAD (%d), UNK (%d), EOS (%d), BOS (%d)' % (self.pad,
                                                                                      self.unk,
                                                                                      self.eos,
                                                                                      self.bos))


    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_src_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
        len_batch = []
        pointer = 'labels_%s' % split
        len_pointer = 'lengths_%s' % split
        max_index = len(self.h5_file[pointer])
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                print('Wrapped source corpus')
                wrapped = True
            self.iterators[split] = ri_next
            label_batch[i] = self.h5_file[pointer][ri, :self.seq_length]
            len_batch.append(min(self.h5_file[len_pointer][ri],
                                 self.seq_length))

        order = sorted(range(batch_size), key=lambda k: -len_batch[k])
        # order = sorted(range(len(x)), key=lambda k: -len(x[k]))  # x : N, T

        data = {}
        data['labels'] = torch.from_numpy(
            label_batch[order, :max(len_batch)]
        ).cuda()

        data['lengths'] = torch.from_numpy(
            np.array([len_batch[k] for k in order]).astype(int)
        ).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data, order

    def get_trg_batch(self, split, order, batch_size=None):
        batch_size = batch_size or self.batch_size
        in_label_batch = np.zeros([batch_size, self.seq_length + 1], dtype='int')
        out_label_batch = np.zeros([batch_size, self.seq_length + 1], dtype='int')
        len_batch = []
        pointer = 'labels_%s' % split
        len_pointer = 'lengths_%s' % split
        max_index = len(self.h5_file[pointer])
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                print('Wrapped target corpus')
                wrapped = True
            self.iterators[split] = ri_next
            # add <bos>
            in_label_batch[i, 0] = self.bos
            in_label_batch[i, 1:] = self.h5_file[pointer][ri, :self.seq_length]
            # add <eos>
            line = self.h5_file[pointer][ri, :self.seq_length]
            ll = min(self.seq_length, self.h5_file[len_pointer][ri])
            len_batch.append(ll + 1)
            out_label_batch[i] = np.insert(line, ll, self.eos)

        data = {}
        # FIXME firs runs were padded till max_length 50
        data['labels'] = torch.from_numpy(in_label_batch[order, :max(len_batch)]).cuda()
        data['out_labels'] = torch.from_numpy(out_label_batch[order, :max(len_batch)]).cuda()
        data['lengths'] = torch.from_numpy(
            np.array([len_batch[k] for k in order]).astype(int)
        ).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

