from .dataloader import textDataLoader
from .split_dataloader import SplitLoader


def ReadData(params, job_name):
    ddir = params['dir']
    src = params['src']
    trg = params['trg']
    dataparams = {'h5': "%s/%s.%s" % (ddir, src, "h5"),
                  'infos': "%s/%s.%s" % (ddir, src, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_src_length']
                  }
    src_loader = textDataLoader(dataparams, job_name=job_name)

    dataparams = {'h5': "%s/%s.%s" % (ddir, trg, "h5"),
                  'infos': "%s/%s.%s" % (ddir, trg, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_trg_length']
                  }

    trg_loader = textDataLoader(dataparams, job_name=job_name)
    return src_loader, trg_loader


def ReadDataSplit(params, split, job_name):
    ddir = params['dir']
    src = params['src']
    trg = params['trg']
    dataparams = {'h5': "%s/%s_%s.%s" % (ddir, src, split, "h5"),
                  'infos': "%s/%s.%s" % (ddir, src, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_src_length']
                  }

    src_loader = SplitLoader(dataparams, job_name=job_name)

    dataparams = {'h5': "%s/%s_%s.%s" % (ddir, trg, split, "h5"),
                  'infos': "%s/%s.%s" % (ddir, trg, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_trg_length']
                  }

    trg_loader = SplitLoader(dataparams, job_name=job_name)
    return src_loader, trg_loader

