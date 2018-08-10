# -*- coding: utf-8 -*-

import os.path as osp
import copy
import pickle
import torch
# from nltk.tokenize.moses import MosesDetokenizer
# DETOK = MosesDetokenizer()
_OKGREEN = '\033[92m'
_WARNING = '\033[93m'
_FAIL = '\033[91m'
_ENDC = '\033[0m'


def _print_sampled(source, gt, pred, score=None):
    transition = '\n>> ' if not score else "\n>> %.3f >> " % score
    source = " ".join(DETOK.detokenize(source.split())).encode('utf-8')
    gt = " ".join(DETOK.detokenize(gt.split())).encode('utf-8')
    pred = " ".join(DETOK.detokenize(pred.split())).encode('utf-8')
    print(source, _OKGREEN, '\nGT:', gt, _WARNING, transition, pred, '\n', _ENDC)
    return pred


def print_sampled(source, gt, pred, score=None):
    transition = '\n>> ' if not score else "\n>> %.3f >> " % score
    source = " ".join(source.split()).encode('utf-8')
    gt = " ".join(gt.split()).encode('utf-8')
    pred = " ".join(pred.split()).encode('utf-8')
    print(source, _OKGREEN, '\nGT:', gt, _WARNING, transition, pred, '\n', _ENDC)
    return pred



def log_epoch(writer, iteration, opt,
              losses, stats, grad_norm,
              ss_prob):

    train_loss = losses['train_loss']
    train_ml_loss = losses['train_ml_loss']
    add_summary_value(writer, 'train/loss', train_loss, iteration)
    add_summary_value(writer, 'train/ml_loss', train_ml_loss, iteration)
    add_summary_value(writer, 'learning_rate', opt.current_lr, iteration)
    add_summary_value(writer, 'scheduled_sampling_prob', ss_prob, iteration)
    add_summary_value(writer, 'RNN_grad_norm', grad_norm, iteration)
    if stats:
        for k in stats:
            add_summary_value(writer, k, stats[k], iteration)
    try:
        train_kld_loss = losses['train_kld_loss']
        train_recon_loss = losses['train/recon_loss']
        add_summary_value(writer, 'train/kld_loss', train_kld_loss, iteration)
        add_summary_value(writer, 'train/recon_loss', train_recon_loss, iteration)
    except:
        pass
    writer.file_writer.flush()  # TF: writer.flush()

def save_model(model, optimizer, opt,
               iteration, epoch, src_loader, trg_loader,
               best_val_score,
               history, best_flag):
    checkpoint_path = osp.join(opt.modelname, 'model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    opt.logger.info("model saved to {}".format(checkpoint_path))
    optimizer_path = osp.join(opt.modelname, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    infos = {}
    # Dump miscalleous informations
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['src_iterators'] = src_loader.iterators
    infos['trg_iterators'] = trg_loader.iterators
    infos['best_val_score'] = best_val_score
    infos['opt'] = copy.copy(opt)
    infos['opt'].logger = None
    infos['val_result_history'] = history['val_perf']
    infos['loss_history'] = history['loss']
    infos['lr_history'] = history['lr']
    infos['scores_stats'] = history['scores_stats']
    infos['ss_prob_history'] = history['ss_prob']
    infos['src_vocab'] = src_loader.get_vocab()
    infos['trg_vocab'] = trg_loader.get_vocab()

    with open(osp.join(opt.modelname, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    if best_flag:
        checkpoint_path = osp.join(opt.modelname, 'model-best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        opt.logger.info("model saved to {}".format(checkpoint_path))
        optimizer_path = osp.join(opt.modelname, 'optimizer-best.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(osp.join(opt.modelname, 'infos-best.pkl'), 'wb') as f:
            pickle.dump(infos, f)


def stderr_epoch(epoch, iteration, opt, losses, grad_norm, ttt):
    train_loss = losses['train_loss']
    train_ml_loss = losses['train_ml_loss']
    message = "iter {} (epoch {}), train_ml_loss = {:.3f}, train_loss = {:.3f}, lr = {:.2e}, grad_norm = {:.3e}"\
               .format(iteration, epoch, train_ml_loss,  train_loss, opt.current_lr, grad_norm)

    try:
        train_kld_loss = losses['train_kld_loss']
        train_recon_loss = losses['train_recon_loss']
        message += "\n{:>25s} = {:.3e}, kld loss = {:.3e}".format('recon loss', train_recon_loss, train_kld_loss)
    except:
        pass
    message += "\n{:>25s} = {:.3f}" \
                .format("Time/batch", ttt)
    opt.logger.debug(message)


def log_optimizer(opt, optimizer):
    opt.logger.debug('########### OPTIMIZER ###########')
    for p in optimizer.param_groups:
        if isinstance(p, dict):
            print('LR:', p['lr'], )
            for pp in p['params']:
                print(pp.size(), end=' ')
            print('\n')
    opt.logger.debug('########### OPTIMIZER ###########')


def add_summary_value_old(writer, key, value, iteration, collections=None):
    """
    Add value to tensorboard events
    """
    # import tensorflow as tf
    _summary = tf.summary.scalar(name=key,
                                 tensor=tf.Variable(value),
                                 collections=collections)
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def add_summary_value(writer, key, value, iteration):
    """
    Add value to tensorboard events
    """
    writer.add_scalar(key, value, iteration)


def save_ens_model(ens_model, optimizer, opt,
                   iteration, epoch, loader, best_val_score,
                   history, best_flag):

    for e, (cnn_model, model) in enumerate(zip(ens_model.cnn_models, ens_model.models)):
        checkpoint_path = osp.join(opt.ensemblename, 'model_%d.pth' % e)
        cnn_checkpoint_path = osp.join(opt.ensemblename, 'model-cnn_%d.pth' % e)
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
        opt.logger.info("model saved to {}".format(checkpoint_path))
        opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
    optimizer_path = osp.join(opt.ensemblename, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    infos = {}
    # Dump miscalleous informations
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['iterators'] = loader.iterators
    infos['best_val_score'] = best_val_score
    infos['opt'] = copy.copy(opt)
    infos['opt'].logger = None
    infos['val_result_history'] = history['val_perf']
    infos['loss_history'] = history['loss']
    infos['lr_history'] = history['lr']
    infos['ss_prob_history'] = history['ss_prob']
    infos['vocab'] = loader.get_vocab()
    with open(osp.join(opt.ensemblename, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    if best_flag:
        for e, (cnn_model, model) in enumerate(zip(ens_model.cnn_models, ens_model.models)):
            checkpoint_path = osp.join(opt.ensemblename, 'model-best_%d.pth' % e)
            cnn_checkpoint_path = osp.join(opt.ensemblename, 'model-cnn-best_%d.pth' % e)
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
            opt.logger.info("model saved to {}".format(checkpoint_path))
            opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
        optimizer_path = osp.join(opt.ensemblename, 'optimizer-best.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(osp.join(opt.ensemblename, 'infos-best.pkl'), 'wb') as f:
            pickle.dump(infos, f)


