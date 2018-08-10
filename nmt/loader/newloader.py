import logging
import pickle
from numbers import Number
import numpy as np
import h5py
import torch
import torch.utils.data as data


def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class H5Dataset(data.Dataset):

    def __init__(self, data_path, split):
        super(H5Dataset, self).__init__()
        self.h5_file = h5py.File('%s_%s.h5' % (data_path, split))
        infos = pickle.load(open('%s.infos' % data_path, 'rb'))
        self.ix_to_word = infos['itow']
        self.vocab_size = len(self.ix_to_word)
        word_to_ix = {v: k for k, v in self.ix_to_word.items()}
        print('in ', data_path, [self.ix_to_word[i] for i in range(4)])
        # Special tokens:
        self.pad_idx = word_to_ix['<PAD>']
        self.bos_idx = word_to_ix['<BOS>']
        self.eos_idx = word_to_ix['<EOS>']
        self.unk_idx = word_to_ix['<UNK>']
        self.labels = self.h5_file['labels']
        self.lengths = self.h5_file['lengths']

    def __getitem__(self, index):
        return torch.from_numpy(self.labels[index].astype(int))

    def __len__(self):
        return self.labels.shape[0]


class LanguagePairDataset(data.Dataset):

    def __init__(self, jobname, params, split):
        self.logger = logging.getLogger(jobname)
        src_path = '%s/%s' % (params['dir'], params['src'])
        trg_path = '%s/%s' % (params['dir'], params['trg'])
        self.src = H5Dataset(src_path, split)
        self.trg = H5Dataset(trg_path, split)
        assert self.src.pad_idx == self.trg.pad_idx
        assert self.src.eos_idx == self.trg.eos_idx
        assert self.src.pad_idx == self.trg.pad_idx
        self.bos_idx = self.trg.bos_idx
        self.eos_idx = self.trg.eos_idx
        self.pad_idx = self.trg.pad_idx
        self.unk_idx = self.trg.unk_idx

    def __getitem__(self, i):
        source = self.src[i]
        target = self.trg[i]
        res = {'id': i,
               'source': source,
               'target': target
               }
        return res

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return LanguagePairDataset.collate(samples,
                                           self.pad_idx,
                                           self.eos_idx,
                                           self.bos_idx)

    @staticmethod
    def collate(samples, pad_idx, eos_idx, bos_idx):
        if len(samples) == 0:
            return {}

        def merge(key, add_bos, add_eos):
            return LanguagePairDataset.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, bos_idx, add_bos, add_eos
            )
        id = torch.IntTensor([s['id'] for s in samples])
        src_tokens = merge('source',
                           add_bos=False,
                           add_eos=False)
        # sort by descending source length
        src_lengths = torch.ShortTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)  # FIXME
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        ntokens = None
        target_out = merge('target',
                           add_bos=False,
                           add_eos=True)
        target_in = merge('target',
                          add_bos=True,
                          add_eos=False)
        trg_lengths = torch.ShortTensor([s['target'].numel() for s in samples])
        # follow the source order:
        target_in = target_in.index_select(0, sort_order)
        target_out = target_out.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        return {
            'id': id,
            'ntokens': ntokens,
            'source': src_tokens.cuda(),
            'source_lengths': src_lengths.cuda(),
            'target_in': target_in.cuda(),
            'target_out': target_out.cuda(),
            'target_lengths': trg_lengths.cuda()
        }

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, bos_idx, add_bos, add_eos):
        size = max(v.size(0) for v in values)
        if add_eos:
            size += 1
        if add_bos:
            size += 1

        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, trg):
            if add_bos:
                # print('adding bos', trg.size(), src.size())
                # assert trg.numel() == src.numel() + 1
                trg[0] = bos_idx
                trg[1:len(src)+1] = src

            elif add_eos:
                # print('adding eos:', trg.size(), src.size())
                # assert trg.numel() == src.numel() + 1
                trg[len(src)] = eos_idx
                trg[:len(src)] = src
            else:
                # print('as is:', trg.size(), src.size())
                trg[:len(src)].copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i])
        return res


def _valid_size(src_size, trg_size, max_positions):
    if isinstance(max_positions, Number):
        max_src_positions, max_trg_positions = max_positions, max_positions
    else:
        max_src_positions, max_trg_positions = max_positions
    if src_size < 1 or src_size > max_src_positions:
        return False
    if trg_size < 1 or trg_size > max_trg_positions:
        return False
    return True


def _make_batches(dataset, indices,
                  max_tokens, max_sentences, max_positions,
                  ignore_invalid_inputs=False,
                  allow_different_src_lens=False):
    src = dataset.src
    trg = dataset.trg
    batch = []

    def yield_batch(next_idx, num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        if not allow_different_src_lens and \
                (src.lengths[batch[0]] != src.lengths[next_idx]):
            return True
        return False

    sample_len = 0
    ignored = []
    for idx in map(int, indices):
        src_size = src.lengths[idx]
        trg_size = trg.lengths[idx]
        if not _valid_size(src_size, trg_size, max_positions):
            if ignore_invalid_inputs:
                ignored.append(idx)
                continue
            raise Exception((
                "Sample #{} has size (src={}, trg={}) but max size is {}."
                " Skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(idx, src_size, trg_size, max_positions))

        sample_len = max(sample_len, src_size, trg_size)
        num_tokens = (len(batch) + 1) * sample_len
        if yield_batch(idx, num_tokens):
            yield batch
            batch = []
            sample_len = max(src_size, trg_size)

        batch.append(idx)

    if len(batch) > 0:
        yield batch

    if len(ignored) > 0:
        print(
            "Warning! {} samples are either too short or too long "
            "and will be ignored, first few sample ids={}".format(
                len(ignored), ignored[:10]
            ))


def batches_as_is(dataset, max_tokens=None,
                  max_sentences=None,
                  max_positions=(1024, 1024)):
    """Returns batches of indices in the loaded order """
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')

    indices = np.arange(len(dataset.src))
    print('indices:', len(indices))
    batches = list(_make_batches(
        dataset, indices, max_tokens,
        max_sentences, max_positions,
        ignore_invalid_inputs=True,
        allow_different_src_lens=True))
    print('starting with:', batches[0])

    return batches


def dataloader(dataset, max_tokens=None,
               max_sentences=None,
               max_positions=(1024, 1024)):
        batch_sampler = batches_as_is(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions
                )
        return data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler
            )



