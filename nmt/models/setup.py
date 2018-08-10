"""
Setup the model and the loss criterion
"""
import nmt.loss as loss
from .seq2seq import Seq2Seq
from .seq2seq_parallel import Seq2Seq_Parallel
from .convs2s2D import Convs2s2D
from .convs2s2D_parallel import Convs2s2D_Parallel



def build_model(jobname, params, src_vocab_size, trg_vocab_size, trg_specials):
    ref = params['model']
    if ref == "seq2seq-attention":
        model = Seq2Seq(jobname, params,
                        src_vocab_size,
                        trg_vocab_size,
                        trg_specials)
    elif ref == "seq2seq-attention-parallel":
        model = Seq2Seq_Parallel(jobname, params,
                                 src_vocab_size,
                                 trg_vocab_size,
                                 trg_specials)

    elif ref == "convs2s2D":
        model = Convs2s2D(jobname, params, src_vocab_size,
                          trg_vocab_size, trg_specials)
    elif ref == "convs2s2D-parallel":
        model = Convs2s2D_Parallel(jobname, params, src_vocab_size,
                                   trg_vocab_size, trg_specials)

    else:
        raise ValueError('Unknown model %s' % ref)

    model.init_weights()
    return model


def define_loss(jobname, params, trg_dict):
    """
    Define training criterion
    """
    ver = params['version'].lower()
    if ver == 'ml':
        crit = loss.MLCriterion(jobname, params)
    elif ver == 'word':
        crit = loss.WordSmoothCriterion(jobname, params)
    elif ver == "seq":
        if params.stratify_reward:
            crit = loss.RewardSampler(jobname, params, trg_dict)
        else:
            crit = loss.ImportanceSampler(jobname, params, trg_dict)
    else:
        raise ValueError('unknown loss mode %s' % ver)
    crit.log()
    return crit


