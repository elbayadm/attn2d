"""
Read config files and pass dict of parameters
"""
import os
import os.path as osp
import argparse
import yaml
import collections
import logging


def read_list(param):
    param = str(param)
    param = [int(p) for p in param.split(',')]
    return param


def update(d, u):
    """update dict of dicts"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ColorStreamHandler(logging.StreamHandler):
    """Logging with colors"""
    DEFAULT = '\x1b[0m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    CYAN = '\x1b[36m'

    CRITICAL = RED
    ERROR = RED
    WARNING = YELLOW
    INFO = GREEN
    DEBUG = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:
            return cls.CRITICAL
        if level >= logging.ERROR:
            return cls.ERROR
        if level >= logging.WARNING:
            return cls.WARNING
        if level >= logging.INFO:
            return cls.INFO
        if level >= logging.DEBUG:
            return cls.DEBUG
        return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return color + text + self.DEFAULT


def create_logger(job_name, log_file=None, debug=True):
    """
    Initialize global logger and return it.
    log_file: log to this file, besides console output
    return: created logger
    TODO: optimize the use of the levels in your logging
    """
    logging.basicConfig(level=5,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.root.handlers = []
    if debug:
        chosen_level = 5
    else:
        chosen_level = logging.INFO
    logger = logging.getLogger(job_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%m/%d %H:%M:%S')
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
        # cerate file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(chosen_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Colored stream handler
    sh = ColorStreamHandler()
    sh.setLevel(chosen_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def parse_eval_params():
    """
    Parse parametres from config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()
    default_config_path = "config/default.yaml"
    if args.config:
        config = yaml.load(open(args.config))
        default_config_path = config.get('default_config', default_config_path)

    default_config = yaml.load(open(default_config_path))
    default_config = update(default_config, config)
    parser.set_defaults(**default_config)
    # Command line arguments - easy access
    parser.add_argument("-v", "--verbose", type=int,
                        default=1, help="code verbosity")
    parser.add_argument("-g", "--gpu_id", type=str,
                        default='0', help="gpu id")
    # only for Evaluation
    parser.add_argument("-b", "--beam_size", type=int,
                        default=1, help="beam size for decoding")
    parser.add_argument("-o", "--offset", type=int,
                        default=0, help="starting index used to visualize a specific batch")
    parser.add_argument("--read_length",  type=int,
                        default=0, help="max length for loading")
    parser.add_argument("--max_length_a",  type=float,
                        default=0, help="beam size for decoding")
    parser.add_argument("--max_length_b",  type=float,
                        default=50, help="beam size for decoding")
    parser.add_argument("-n", "--batch_size", type=int,
                        default=5, help="batch size for decoding")
    parser.add_argument("-l", "--last", action="store_true")
    parser.add_argument("--norm", action="store_true", help="Normalize scores by length")
    parser.add_argument("--edunov", action="store_true", help="eval on edunov's test")
    parser.add_argument("-m", "--max_samples", type=int,
                        default=100)
    parser.add_argument("--block_ngram_repeat", type=int,
                        default=0)
    parser.add_argument("--length_penalty", "-p", type=float,
                        default=0.6, help="length penalty for GNMTscorer")

    parser.add_argument("--length_penalty_mode", type=str,
                        default="wu", help="length penalty mode, either wu or avg for GNMTscorer")

    parser.add_argument("--stepwise_penalty", action="store_true")
    parser.add_argument("-s", "--split", type=str,
                        default="test", help="Evaluation split")

    args = parser.parse_args(remaining_argv)
    # mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not osp.exists(args.eventname):
        os.makedirs(args.eventname)
    if not osp.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    logger = create_logger(args.modelname+"_eval", '%s/eval.log' % args.modelname)
    args = vars(args)
    # parse list params #TODO check syntax for yaml directly
    if "network" in args:
        if "num_layers" in args['network']:
            num_layers = read_list(args['network']['num_layers'])
            if "kernels" in args['network']:
                kernels = read_list(args['network']['kernels'])
                if len(kernels) == 1:
                    args['network']['kernels'] = kernels * len(num_layers)
                else:
                    assert len(kernels) == len(num_layers), "the number of kernel sizes must match that of the network layers"
                    args["network"]["kernels"] = kernels

            args['network']['num_layers'] = num_layers

    return args


def parse_params():
    """
    Parse parametres from config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()
    default_config_path = "config/default.yaml"
    if args.config:
        config = yaml.load(open(args.config))
        default_config_path = config.get('default_config', default_config_path)

    default_config = yaml.load(open(default_config_path))
    default_config = update(default_config, config)
    parser.set_defaults(**default_config)
    # Command line arguments - easy access
    parser.add_argument("-v", "--verbose", type=int,
                        default=1, help="code verbosity")
    parser.add_argument("-g", "--gpu_id", type=str,
                        default='0', help="gpu id")

    args = parser.parse_args(remaining_argv)
    # mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not osp.exists(args.eventname):
        os.makedirs(args.eventname)
    if not osp.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    logger = create_logger(args.modelname, '%s/train.log' % args.modelname)
    args = vars(args)
    # parse list params #TODO check syntax for yaml directly
    if "network" in args:
        if "num_layers" in args['network']:
            num_layers = read_list(args['network']['num_layers'])
            if "kernels" in args['network']:
                kernels = read_list(args['network']['kernels'])
                if len(kernels) == 1:
                    args['network']['kernels'] = kernels * len(num_layers)
                else:
                    assert len(kernels) == len(num_layers), "the number of kernel sizes must match that of the network layers"
                    args["network"]["kernels"] = kernels

            args['network']['num_layers'] = num_layers

    return args

