import sys
import subprocess
import json
import random
import time
import logging
import os.path as osp
import numpy as np
import torch
import gc
from tensorboardX import SummaryWriter

import torch
from nmt.utils import pl, pd
from nmt.models.evaluate import evaluate_model, evaluate_val_loss, evaluate_split
from nmt.loader import ReadDataSplit
from nmt.loader.pair_loader import DataPair

from .optimizer import Optimizer,  LRScheduler
from ._trackers import TRACKERS


class Trainer(object):
    """
    Training a model with a given criterion
    """

    def __init__(self, jobname, params, model, criterion):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.params = params
        self.jobname = jobname

        self.logger = logging.getLogger(jobname)
        # reproducibility:
        seed = params['optim']['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.clip_norm = params['optim']['grad_clip']
        self.num_batches = params['optim']['num_batches']

        # copy model and criterion to current device
        # FIXME before or after loading
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        # initialize optimizer and LR scheduler
        self.optimizer = Optimizer(params['optim'], model)
        self.lr_patient = params['optim']['LR']['schedule'] == "early-stopping"
        if self.lr_patient:
            self.lr_patient = params['optim']['LR']['criterion']
            self.logger.info('updating the lr wrt %s', self.lr_patient)
        self.lr_scheduler = LRScheduler(params['optim']['LR'],
                                        self.optimizer.optimizer,
                                        )
        # self.ss_scheduler = LRScheduler(params['optim'], self.optimizer.optimizer)
        # self.alpha_scheduler = LRScheduler(params['optim'], self.optimizer.optimizer)

        self.tb_writer = SummaryWriter(params['eventname'])
        self.log_every = params['track']['log_every']
        self.checkpoint = params['track']['checkpoint']
        self.evaluate = False
        self.done = False
        self.trackers = TRACKERS
        self.iteration = 0
        self.epoch = 0
        self.batch_offset = 0
        # Dump  the model params:
        json.dump(params, open('%s/params.json' % params['modelname'], 'w'))

    def track(self, k, v):
        if k not in self.trackers:
            raise ValueError('Tracking unknown entity %s' % k)
        if isinstance(self.trackers[k], list):
            self.trackers[k].append(v)
        else:
            self.trackers[k] = v
        self.trackers['update'].add(k)

    def log(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def load_for_eval(self, best=True):
        # Restart training (useful with oar idempotant)
        params = self.params
        modelname = params['modelname']
        iterators_state = {}
        history = {}
        if best:
            flag = '-best'
        else:
            flag = ''
        modelpath = osp.join(modelname, "model%s.pth" % flag)
        # load model's weights
        self.model.load_state_dict(
            torch.load(modelpath)
        )
        return modelpath


    def load_checkpoint(self):
        # Restart training (useful with oar idempotant)
        params = self.params
        modelname = params['modelname']
        iterators_state = {}
        history = {}
        if osp.exists(osp.join(modelname, 'model.pth')):
            self.warn('Picking up where we left')
            # load model's weights
            saved_state = torch.load(osp.join(modelname, 'model.pth'))
            saved = list(saved_state)
            required_state = self.model.state_dict()
            required = list(required_state)
            del required_state
            if "module" in required[0] and "module" not in saved[0]:
                for k in saved:
                    kbis = "module.%s" % k
                    saved_state[kbis] = saved_state[k]
                    del saved_state[k]

            for k in saved:
                if "increment" in k:
                    del saved_state[k]
            self.model.load_state_dict(saved_state)
            # load the optimizer's last state:
            self.optimizer.load(
                torch.load(osp.join(modelname, 'optimizer.pth')
                           ))
            history = pl(osp.join(modelname, 'trackers.pkl'))
            iterators_state = {'src_iterators': history['src_iterators'],
                               'trg_iterators': history['trg_iterators']}

        elif params['start_from']:
            start_from = params['start_from']
            # Start from a pre-trained model:
            self.warn('Starting from %s' % start_from)
            if params['start_from_best']:
                flag = '-best'
                self.warn('Starting from the best saved model')
            else:
                flag = ''
            # load model's weights
            saved_weights = torch.load(osp.join(start_from, 'model%s.pth' % flag))
            if params['reverse']:
                # adapat the loaded weight for the reverse task
                saved_weights = self.reverse_weights(saved_weights)
            self.model.load_state_dict(saved_weights)
            del saved_weights
            # load the optimizer's last state:
            if not params['optim']['reset']:
                self.optimizer.load(
                    torch.load(osp.join(start_from, 'optimizer%s.pth' % flag)
                               ))
            history = pl(osp.join(start_from, 'trackers%s.pkl' % flag))
        self.trackers.update(history)
        self.epoch = self.trackers['epoch']
        self.iteration = self.trackers['iteration']
        # start with eval:
        # self.iteration -= 1
        return iterators_state

    def reverse_weights(self, loaded):
        for k in loaded:
            print(k, loaded[k].size())
        current = self.model.state_dict()
        for k in current:
            print(k, current[k].size())
        return loaded

    def update_params(self, val_loss=None):
        """
        Update dynamic params: lr, scheduled_sampling probability and tok/seq's alpha
        """
        epoch = self.epoch
        iteration = self.iteration
        if not self.lr_patient:
            if self.lr_scheduler.mode in ["step-iter", "inverse-square"]:
                self.lr_scheduler.step(iteration)
            else:
                self.lr_scheduler.step(epoch - 1)
        self.track('optim/lr', self.optimizer.get_lr())
        # Assign the scheduled sampling prob
        # self.model.ss_prob = self.ss_scheduler.step(epoch, iteration)
        # if self.criterion.version in ['tok', 'seq']:
            # self.criterion.alpha = self.alpha_scheduler.step(epoch, iteration)

    def step(self, data_src, data_trg, ntokens=0):
        # self.log('max length: %d' % data_src['labels'].size(1))
        # average multiple batches before the backward:
        # Clear the grads
        self.optimizer.zero_grad()
        start = time.time()
        batch_size = data_src['labels'].size(0)
        # self.debug('source %s' % str(data_src['labels'].size()))
        # self.debug('target %s' % str(data_trg['labels'].size()))

        # start = time.time()
        # evaluate the loss
        if self.criterion.version == "seq":
            source = self.model.encoder(data_src)
            source = self.model.map(source)
            losses, stats = self.criterion(self.model, source, data_trg)

        else:  # ML & Token-level
            # init and forward decoder combined
            decoder_logit = self.model(data_src, data_trg)
            losses, stats = self.criterion(decoder_logit, data_trg['out_labels'])

        if not ntokens:
            ntokens = torch.sum(data_src['lengths'] *
                                data_trg['lengths']).data.item()

        return losses, batch_size, ntokens

    def backward_step(self, loss, ml_loss, ntokens, nseqs, start, wrapped):
        # losses['final'].backward()
        loss.backward()

        if self.clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.clip_norm)
        self.track('optim/grad_norm', grad_norm)
        self.track('optim/ntokens', ntokens)
        self.track('optim/batch_size', nseqs)

        # self.track('optim/grad_norm', self.optimizer.clip_gradient())
        self.optimizer.step()
        # torch.cuda.empty_cache()  # FIXME choose an optimal freq
        gc.collect()
        if np.isnan(loss.data.item()):
            sys.exit('Loss is nan')
        torch.cuda.synchronize()
        self.iteration += 1
        # wrapped = data_src['bounds']['wrapped']:
        if wrapped:
            self.epoch += 1
        # Write the training loss summary
        if (self.iteration % self.log_every == 0):
            self.track('train/loss', loss.data.item())
            self.track('train/ml_loss', ml_loss.data.item())
            self.to_stderr(nseqs, ntokens, time.time()-start)
            self.tensorboard()

        self.evaluate = (self.iteration % self.checkpoint == 0)
        self.done = (self.epoch > self.params['optim']['max_epochs'])

    def tensorboard(self):
        for k in self.trackers['update']:
            self.tb_writer.add_scalar(k, self.trackers[k][-1], self.iteration)
        self.tb_writer.file_writer.flush()
        self.trackers['update'] = set()

    def to_stderr(self, batch_size, ntokens, timing):
        self.log('| epoch {:2d} '
                 '| iteration {:5d} '
                 '| lr {:02.2e} '
                 '| tokens({:3d}) {:5d} '
                 '| ms/batch {:6.3f} '
                 '| loss {:6.3f} '
                 '| ml {:6.3f}'
                 .format(self.epoch,
                         self.iteration,
                         self.optimizer.get_lr(),
                         batch_size,
                         ntokens,
                         timing * 1000,
                         self.trackers['train/loss'][-1],
                         self.trackers['train/ml_loss'][-1]))

    def validate(self, src_loader=None, trg_loader=None):
        """
        Score the validation set
        """
        params = self.params
        self.log('Evaluating the model on the validation set..')
        self.model.eval()

        if src_loader is None:
            params["data"]['max_tokens'] = params['track']['max_tokens']
            params["data"]['max_batch_size'] = params['track']['max_batch_size']

            loader = DataPair(params['data'], 'val', self.jobname)
            src_loader = loader.src
            trg_loader = loader.trg
            _, val_ml_loss, val_loss, bleu = evaluate_split(self.jobname,
                                                            self,
                                                            loader,
                                                            params['track'])
            self.log('BLEU: %.5f ' % bleu)
            self.track('val/perf/bleu', bleu)
            save_best = (self.trackers['val/perf/bleu'][-1] ==
                         max(self.trackers['val/perf/bleu']))
            save_every = 0

        else:
            if params.get('eval_bleu', 1):
                _, val_ml_loss, val_loss, bleu = evaluate_model(params['modelname'],
                                                                self,
                                                                src_loader,
                                                                trg_loader,
                                                                params['track'])
                self.log('BLEU: %.5f ' % bleu)
                self.track('val/perf/bleu', bleu)
                save_best = (self.trackers['val/perf/bleu'][-1] ==
                             max(self.trackers['val/perf/bleu']))
                save_every = 0

            else:
                val_ml_loss, val_loss = evaluate_val_loss(params['modelname'],
                                                          self,
                                                          src_loader,
                                                          trg_loader,
                                                          params['track'])
                save_every = 1
                save_best = 0

        # Write validation result into summary
        self.track('val/loss', val_loss)
        self.track('val/ml_loss', val_ml_loss)
        self.tensorboard()
        # Save model if is improving on validation result
        self.save_model(src_loader, trg_loader, save_best, save_every)
        self.model.train()
        if self.lr_patient == "loss":
            self.log('Updating the learning rate - LOSS')
            self.lr_scheduler.step(val_loss)
            self.track('optim/lr', self.optimizer.get_lr())
        elif self.lr_patient == "perf":
            assert not save_every
            self.log('Updating the learning rate - PERF')
            self.lr_scheduler.step(bleu)
            self.track('optim/lr', self.optimizer.get_lr())

    def save_model(self, src_loader, trg_loader, save_best, save_every):
        """
        checkoint model, optimizer and misc
        """
        params = self.params
        modelname = params['modelname']
        checkpoint_path = osp.join(modelname, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        self.log("model saved to {}".format(checkpoint_path))
        optimizer_path = osp.join(modelname, 'optimizer.pth')
        torch.save(self.optimizer.state_dict(), optimizer_path)
        self.log("optimizer saved to {}".format(optimizer_path))
        self.trackers['src_iterators'] = src_loader.iterators
        self.trackers['trg_iterators'] = trg_loader.iterators
        self.trackers['iteration'] = self.iteration
        self.trackers['epoch'] = self.epoch
        pd(self.trackers, osp.join(modelname, 'trackers.pkl'))

        if save_best:
            checkpoint_path = osp.join(modelname, 'model-best.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log("model saved to {}".format(checkpoint_path))
            optimizer_path = osp.join(modelname, 'optimizer-best.pth')
            torch.save(self.optimizer.state_dict(), optimizer_path)
            self.log("optimizer saved to {}".format(optimizer_path))
            pd(self.trackers, osp.join(modelname, 'trackers-best.pkl'))

        if save_every:
            checkpoint_path = osp.join(modelname, 'model-%d.pth' % self.iteration)
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log("model saved to {}".format(checkpoint_path))
