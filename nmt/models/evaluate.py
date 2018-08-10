# -*- coding: utf-8 -*-

"""Evaluation utils."""
import gc
import logging
from collections import Counter
import math
import time
import subprocess
import numpy as np
import torch
from torch.autograd import Variable
from nmt.utils import decode_sequence
import nmt.utils.logging as lg
from nmt.models.gnmt import GNMTGlobalScorer

# ESKE
def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a list of translation hypotheses and references.
    With the default settings, this computes the exact same score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n. `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((order,))
    correct = np.zeros((order,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        hyp_length += len(hyp)
        ref_length += len(ref)

        for i in range(order):
            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))
            ref_ngrams = Counter(zip(*[ref[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    def divide(x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.true_divide(x, y)
            z[~ np.isfinite(z)] = 0
        return z

    scores = divide(correct, total)

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length)) if hyp_length > 0 else 0.0
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""

    def bleu_stats(hypothesis, reference):
        """Compute statistics for BLEU."""
        stats = []
        stats.append(len(hypothesis))
        stats.append(len(reference))
        for n in range(1, 5):
            s_ngrams = Counter(
                [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
            )
            r_ngrams = Counter(
                [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
            )
            stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
            stats.append(max([len(hypothesis) + 1 - n, 0]))
        return stats

    def bleu(stats):
        """Compute BLEU given n-gram statistics."""
        if len([x for x in stats if x == 0]) > 0:
            return 0
        (c, r) = stats[:2]
        log_bleu_prec = sum(
            [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
        ) / 4.
        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", 'multi-bleu.perl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()


def model_perplexity(model, src_loader, trg_loader, split="val", logger=None):
    """Compute model perplexity."""
    # Make sure to be in evaluation mode
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    loss_evals = 0
    while True:
        # get batch
        data_src = src_loader.get_src_batch(split)
        input_lines_src = data_src['labels']
        input_lines_src = Variable(torch.from_numpy(input_lines_src),
                                   requires_grad=False).cuda()

        data_trg = trg_loader.get_trg_batch(split)
        tmp = [data_trg['labels'], data_trg['out_labels'], data_trg['mask']]
        input_lines_trg, output_lines_trg, mask = [Variable(torch.from_numpy(_),
                                                            requires_grad=False).cuda()
                                                   for _ in tmp]

        n = n + src_loader.batch_size
        decoder_logit = model(input_lines_src, input_lines_trg)
        ml_loss, loss, stats = model.crit(decoder_logit, output_lines_trg, mask)
        loss_sum = loss_sum + loss.data.item()
        loss_evals = loss_evals + 1

        ix1 = data_src['bounds']['it_max']
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    return loss_sum / loss_evals


def evaluate_val_loss(job_name, trainer, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    max_samples = eval_kwargs.get('max_samples', -1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    logger = logging.getLogger(job_name)

    # Make sure to be in evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
    # print('Predictions lenght:', len(preds), len(ground_truths))
    # assert(len(preds) == trg_loader.h5_file['labels_val'].shape[0])
    logger.warn('Evaluated %d samples in %.2f s', n, time.time()-start)
    return ml_loss_sum / loss_evals, loss_sum / loss_evals


def evaluate_loader(job_name, trainer, loader, src_dict, trg_dict, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    max_samples = eval_kwargs.get('max_samples', math.inf)
    verbose = eval_kwargs.get('verbose', 0)
    logger = logging.getLogger(job_name)

    # Make sure to be in evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    for i, sample in enumerate(loader, start=0):
        # get batch
        data_src = {"labels": sample['net_input']['src_tokens'].cuda(),
                    "lengths": sample['net_input']['src_lengths'].cuda()
                    }
        data_trg = {"labels": sample['net_input']['prev_output_tokens'].cuda(),
                    "out_labels": sample['target'].cuda(),
                    "lengths": sample['net_input']['src_lengths'].cuda()  # modify loader to return trg lengths as well TODO
                    }

        del sample
        # print("batch:", data_src['labels'].size(0))
        n += data_src['labels'].size(0)
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
            batch_preds, _ = model.sample(source, eval_kwargs)
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])
            batch_preds, _ = model.sample(data_src, eval_kwargs)

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        torch.cuda.empty_cache()  # FIXME choose an optimal freq
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_dict,
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_dict.eos(),
                                          bos=trg_dict.eos())[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_dict, batch_preds,
                                         eos=trg_dict.eos(),
                                         bos=trg_dict.eos())
        # Do the same for gold sentences
        sent_source = decode_sequence(src_dict,
                                      data_src['labels'],
                                      eos=src_dict.eos(),
                                      bos=src_dict.eos()
                                      )
        sent_gold = decode_sequence(trg_dict,
                                    data_trg['out_labels'],
                                    eos=trg_dict.eos(),
                                    bos=trg_dict.eos()
                                    )
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if n > max_samples:
            break
    # print('Predictions lenght:', len(preds), len(ground_truths))
    # assert(len(preds) == trg_loader.h5_file['labels_val'].shape[0])
    logger.warn('Evaluated %d samples in %.2f s', len(preds), time.time()-start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, ml_loss_sum / loss_evals, loss_sum / loss_evals, bleu_moses


def evaluate_split(job_name, trainer, loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    max_samples = eval_kwargs.get('max_samples', -1)
    verbose = eval_kwargs.get('verbose', 0)
    logger = logging.getLogger(job_name)

    src_loader = loader.src
    trg_loader = loader.trg
    # Make sure to be in evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    while True:
        # get batch
        sample = loader.get_batch()
        data_src = sample["src"]
        data_trg = sample["trg"]
        ntokens = sample['ntokens']
        del sample
        print('Eval ntokens:', ntokens, "batch:", data_src['labels'].size(0))
        n += data_src['labels'].size(0)
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
            batch_preds, _ = model.sample(source, eval_kwargs)
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])
            batch_preds, _ = model.sample(data_src, eval_kwargs)

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        torch.cuda.empty_cache()  # FIXME choose an optimal freq
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'],
                                      eos=src_loader.eos,
                                      bos=src_loader.bos)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
    # print('Predictions lenght:', len(preds), len(ground_truths))
    # assert(len(preds) == trg_loader.h5_file['labels_val'].shape[0])
    logger.warn('Evaluated %d samples in %.2f s', len(preds), time.time()-start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, ml_loss_sum / loss_evals, loss_sum / loss_evals, bleu_moses


def evaluate_model(job_name, trainer, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    max_samples = eval_kwargs.get('max_samples', -1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    logger = logging.getLogger(job_name)

    # Make sure to be in evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
            batch_preds, _ = model.sample(source, eval_kwargs)
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])
            batch_preds, _ = model.sample(data_src, eval_kwargs)

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'],
                                      eos=src_loader.eos,
                                      bos=src_loader.bos)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
    # print('Predictions lenght:', len(preds), len(ground_truths))
    # assert(len(preds) == trg_loader.h5_file['labels_val'].shape[0])
    logger.warn('Evaluated %d samples in %.2f s', len(preds), time.time()-start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, ml_loss_sum / loss_evals, loss_sum / loss_evals, bleu_moses


def score_trads(preds, trg_loader, eval_kwargs):
    split = eval_kwargs.get('split', 'val')
    batch_size = eval_kwargs.get('batch_size', 80)
    verbose = eval_kwargs.get('verbose', 0)
    ground_truths = []
    trg_loader.reset_iterator(split)
    n = 0
    while True:
        # get batch
        data_trg = trg_loader.get_trg_batch(split,
                                            range(batch_size),
                                            batch_size)
        output_lines_trg_gold = data_trg['out_labels']
        n += batch_size
        # Decode a minibatch greedily __TODO__ add beam search decoding
        # Do the same for gold sentences
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    output_lines_trg_gold,
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (l, gl) in zip(preds, sent_gold):
            ground_truths.append(gl)
            if verb:
                lg.print_sampled("", gl, l)
        ix1 = data_trg['bounds']['it_max']
        if data_trg['bounds']['wrapped']:
            break
        if n >= ix1:
            print('Evaluated the required samples (%s)' % n)
            break
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    scores = {'Bleu': bleu_moses}
    return scores

def sample_model(job_name, model, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    print('src_loader ref:', src_loader.ref)
    # remove_bpe = 'BPE' in src_loader.ref
    remove_bpe = eval_kwargs.get('remove_bpe', True)
    print('Removing bpe:', remove_bpe)
    logger = logging.getLogger(job_name)
    # Make sure to be in evaluation mode
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    start = time.time()
    lenpen_mode = eval_kwargs.get('lenpen_mode', 'wu')
    scorer = GNMTGlobalScorer(eval_kwargs['lenpen'], 0, 'none', lenpen_mode)

    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            batch_preds, _ = model.decoder.sample(source, scorer, eval_kwargs)
        else:
            batch_preds, _ = model.sample(data_src, scorer, eval_kwargs)

        torch.cuda.empty_cache()  # FIXME choose an optimal freq
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos,
                                          remove_bpe=remove_bpe)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos,
                                         remove_bpe=remove_bpe)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'],
                                      eos=src_loader.eos,
                                      bos=src_loader.bos,
                                      remove_bpe=remove_bpe)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos,
                                    remove_bpe=remove_bpe)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                print('n:', n)
                lg.print_sampled(sl, gl, l)
        ix1 = data_src['bounds']['it_max']
        # ix1 = 20
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
        del sent_source, sent_preds, sent_gold, batch_preds
        gc.collect()
    logger.warn('Sampled %d sentences in %.2f s', len(preds), time.time() - start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, bleu_moses


def track_model(job_name, model, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    source = []
    preds = []
    ground_truths = []
    batched_alphas = []
    batched_aligns = []
    batched_activ_aligns = []
    batched_activs = []
    batched_embed_activs = []
    batch_size = eval_kwargs.get('batch_size', 1)
    assert batch_size == 1, "Batch size must be 1"
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    max_samples = eval_kwargs.get('max_samples', -1)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    print('src_loader ref:', src_loader.ref)
    remove_bpe = 'BPE' in src_loader.ref
    print('Removing bpe:', remove_bpe)
    logger = logging.getLogger(job_name)
    # Make sure to be in evaluation mode
    model.eval()
    offset = eval_kwargs.get('offset', 0)
    print('Starting from ', offset)
    src_loader.iterators[split] = offset
    trg_loader.iterators[split] = offset
    # src_loader.reset_iterator(split)
    # trg_loader.reset_iterator(split)
    n = 0
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            batch_preds, _ = model.decoder.sample(source, eval_kwargs)
        else:
            # track returns seq, alphas, aligns, activ_aligns, activs, embed_activs, clean_cstr
            batch_preds, alphas, aligns, activ_aligns, activs, embed_activs, C = model.track(data_src, eval_kwargs)
            batched_alphas.append(alphas)
            batched_aligns.append(aligns)
            batched_activ_aligns.append(activ_aligns)
            batched_activs.append(activs)
            batched_embed_activs.append(embed_activs)

        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos,
                                          remove_bpe=False)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos,
                                         remove_bpe=False)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'].data.cpu().numpy(),
                                      eos=src_loader.eos,
                                      bos=src_loader.bos,
                                      remove_bpe=False)
        source.append(sent_source)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'].data.cpu().numpy(),
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos,
                                    remove_bpe=False)
        if not verbose:
            verb = not (n % 300)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples

        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    print('Sampled %d sentences' % len(preds))
    bleu_moses, _ = corpus_bleu(preds, ground_truths)

    return {'source': source,
            'preds': preds,
            'alpha': batched_alphas,
            'align': batched_aligns,
            'activ_align': batched_activ_aligns,
            'activ': batched_activs,
            'embed_activ': batched_embed_activs,
            'channels_cst': C,
            "bleu": bleu_moses,
            }
