import sys
import os.path as osp
import argparse
import h5py
import numpy as np
from nmt.utils import pd


def build_vocab(sentences, max_words, vocab_file):
    """
    Build vocabulary
    Note: I use the perl scripts instead
    """
    # count up the number of words
    counts = {}
    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for txt in sentences:
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
        for w in txt:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    vocab = [w for (c, w) in cw[:max_words]]
    bad_words = [w for (c, w) in cw[max_words:]]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    # print('sentence length distribution (count, number of words):')
    # sum_len = sum(sent_lengths.values())
    # for i in range(max_len+1):
        # print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len))

    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.insert(0, "<BOS>")
    vocab.insert(0, "<EOS>")
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<PAD>")
    # writing a vocab file:
    with open(vocab_file, 'w') as fv:
        for word in vocab:
            fv.write(word+'\n')
    # Dump the statistics for later use:
    pd({"counts": counts,
        "vocab": vocab,
        "bad words": bad_words,
        "lengths": sent_lengths},
       vocab_file + ".stats")

    return vocab


def encode_sentences(sentences, params, wtoi):
    """
    encode all sentences into one large array, which will be 1-indexed.
    No special tokens are added, except from the <pad> after the effective length
    """
    max_length = params.max_length
    lengths = []
    m = len(sentences)
    IL = np.zeros((m, max_length), dtype='uint32')  # <PAD> token is 0
    M = np.zeros((m, max_length), dtype='uint32')
    for i, sent in enumerate(sentences):
        lengths.append(len(sent))
        for k, w in enumerate(sent):
            if k < max_length:
                IL[i, k] = wtoi[w] if w in wtoi else wtoi['<UNK>']
                M[i, k] = int(w in wtoi)
        # bar.update(i)
        if not i % 10000:
            print(i)

    assert np.all(np.array(lengths) > 0), 'error: some line has no words'
    return IL, M, lengths


def main_trg(params, train_order, val_order, test_order, vocab=None):
    """
    Main preprocessing
    """
    max_length = params.max_length
    train_trg = 'data/%s/train.%s' % (params.data_dir, params.trg)
    val_trg = 'data/%s/valid.%s' % (params.data_dir, params.trg)
    test_trg = 'data/%s/test.%s' % (params.data_dir, params.trg)
    with open(train_trg, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
        if train_order is not None:
            sentences = [sentences[k] for k in train_order]
    print("Read %d lines from %s" % (len(sentences), train_trg))

    if vocab is None:
        vocab_file = "data/%s/vocab.%s" % (params.data_dir, params.trg)
        if osp.exists(vocab_file):
            # If reading from an existing vocab file
            vocab = []
            for line in open(vocab_file, 'r'):
                vocab.append(line.strip())
            if '<BOS>' not in vocab:
                print('Inserting BOS')
                vocab.insert(0, "<BOS>")
            if '<EOS>' not in vocab:
                print('Inserting EOS')
                vocab.insert(0, "<EOS>")
            if '<UNK>' not in vocab:
                print('Inserting UNK')
                vocab.insert(0, "<UNK>")
            if '<PAD>' not in vocab:
                print('Inserting PAD')
                vocab.insert(0, "<PAD>")
        else:
            # create the vocab
            vocab = build_vocab(sentences, params.max_words_trg,
                                vocab_file)
    print('Length of vocab:', len(vocab))
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    # encode captions in large arrays, ready to ship to hdf5 file
    IL_train, Mask_train, Lengths_train = encode_sentences(sentences, params, wtoi)

    with open(val_trg, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
        if val_order is not None:
            sentences = [sentences[k] for k in val_order]

    print("Read %d lines from %s" % (len(sentences), val_trg))
    IL_val, Mask_val, Lengths_val = encode_sentences(sentences, params, wtoi)

    with open(test_trg, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
        if test_order is not None:
            sentences = [sentences[k] for k in test_order]

    print("Read %d lines from %s" % (len(sentences), test_trg))
    IL_test, Mask_test, Lengths_test = encode_sentences(sentences, params, wtoi)

    # create output h5 file
    f = h5py.File('data/%s/%s.h5' % (params.data_dir, params.trg), "w")
    f.create_dataset("labels_train", dtype='uint32', data=IL_train)
    # f.create_dataset("mask_train", dtype='uint32', data=Mask_train)
    f.create_dataset("lengths_train", dtype='uint32', data=Lengths_train)

    f.create_dataset("labels_val", dtype='uint32', data=IL_val)
    # f.create_dataset("mask_val", dtype='uint32', data=Mask_val)
    f.create_dataset("lengths_val", dtype='uint32', data=Lengths_val)

    f.create_dataset("labels_test", dtype='uint32', data=IL_test)
    # f.create_dataset("mask_test", dtype='uint32', data=Mask_test)
    f.create_dataset("lengths_test", dtype='uint32', data=Lengths_test)

    print('wrote h5file for the target langauge')
    pd({'itow': itow, 'params': params},
       'data/%s/%s.infos' % (params.data_dir, params.trg))


def main_src(params):
    """
    Main preprocessing
    """
    max_length = params.max_length
    train_src = 'data/%s/train.%s' % (params.data_dir, params.src)
    val_src = 'data/%s/valid.%s' % (params.data_dir, params.src)
    test_src = 'data/%s/test.%s' % (params.data_dir, params.src)
    batch_size = 32
    with open(train_src, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
    if params.sort:
        print('Sorting by length')
        train_order = sorted(range(len(sentences)),
                             key=lambda k: len(sentences[k]),
                             reverse=True)
        sentences = [sentences[k] for k in train_order]
    elif params.shuffle_sort:
        print('Sorting by length')
        train_order = sorted(range(len(sentences)),
                             key=lambda k: len(sentences[k]),
                             reverse=True)
        batched_order = np.array_split(train_order,
                                       len(train_order)//batch_size)
        print('Shuffling:')
        np.random.shuffle(batched_order)
        train_order = np.concatenate(batched_order)
        sentences = [sentences[k] for k in train_order]
    else:
        train_order = None

    print("Read %d lines from %s" % (len(sentences), train_src))

    vocab_file = "data/%s/vocab.%s" % (params.data_dir, params.src)
    if osp.exists(vocab_file):
        # If reading from an existing vocab file
        vocab = []
        for line in open(vocab_file, 'r'):
            vocab.append(line.strip())
        if '<BOS>' not in vocab:
            print('Inserting BOS')
            vocab.insert(0, "<BOS>")
        if '<EOS>' not in vocab:
            print('Inserting EOS')
            vocab.insert(0, "<EOS>")
        if '<UNK>' not in vocab:
            vocab.insert(0, "<UNK>")
        if '<PAD>' not in vocab:
            vocab.insert(0, "<PAD>")
    else:
        # create the vocab
        vocab = build_vocab(sentences, params.max_words_src,
                            vocab_file)
    print('Length of vocab:', len(vocab))
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    # encode captions in large arrays, ready to ship to hdf5 file
    IL_train_src, _, Lengths_train = encode_sentences(sentences, params, wtoi)

    with open(val_src, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
    if params.shuffle_sort_eval:
        print('Sorting by length')
        val_order = sorted(range(len(sentences)),
                           key=lambda k: len(sentences[k]),
                           reverse=True)
        batched_order = np.array_split(val_order,
                                       len(val_order)//batch_size)
        print('Shuffling:')
        np.random.shuffle(batched_order)
        val_order = np.concatenate(batched_order)
        sentences = [sentences[k] for k in val_order]
    else:
        val_order = None
    print("Read %d lines from %s" % (len(sentences), val_src))
    IL_val_src, _, Lengths_val = encode_sentences(sentences, params, wtoi)

    with open(test_src, 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:max_length] for sent in sentences]
    test_order = None

    print("Read %d lines from %s" % (len(sentences), test_src))
    IL_test_src, _, Lengths_test = encode_sentences(sentences, params, wtoi)
    # create output h5 file
    f = h5py.File('data/%s/%s.h5' % (params.data_dir, params.src), "w")
    f.create_dataset("labels_train", dtype='uint32', data=IL_train_src)
    f.create_dataset("lengths_train", dtype='uint32', data=Lengths_train)
    f.create_dataset("labels_val", dtype='uint32', data=IL_val_src)
    f.create_dataset("lengths_val", dtype='uint32', data=Lengths_val)
    f.create_dataset("labels_test", dtype='uint32', data=IL_test_src)
    f.create_dataset("lengths_test", dtype='uint32', data=Lengths_test)

    print('wrote h5file for the source langauge')
    pd({'itow': itow, 'params': params},
       'data/%s/%s.infos' % (params.data_dir, params.src))
    return train_order, val_order, test_order, vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='WMT14')
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--trg', type=str, default='fr')
    parser.add_argument('--max_words_src', default=30000, type=int,
                        help="Max words in the source vocabulary")
    parser.add_argument('--max_words_trg', default=30000, type=int,
                        help="Max words in the target vocabulary")
    parser.add_argument('--max_length', default=50, type=int,
                        help='max length of a sentence')
    parser.add_argument('--sort', action='store_true', help='sort the training set by source sequence length')
    parser.add_argument('--share_vocab', action='store_true', help='share the source and target vocab')
    parser.add_argument('--shuffle_sort', action='store_true', help='sort the training set by source sequence length')
    parser.add_argument('--shuffle_sort_eval', action='store_true', help='sort the training set by source sequence length')

    params = parser.parse_args()
    # Default settings:
    if params.data_dir == 'iwslt14':
        params.src = "de"
        params.trg = "en"
        params.max_words_src = 14000
        params.max_words_trg = 14000
        params.shuffle_sort = 1
        params.shuffle_sort_eval = 0
        params.max_length = 200

    train_order, val_order, test_order, vocab = main_src(params)
    if params.share_vocab:
        main_trg(params, train_order, val_order, test_order, vocab)
    else:
        main_trg(params, train_order, val_order, test_order)
