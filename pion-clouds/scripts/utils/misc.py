import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
from nflows import transforms, distributions, flows
import torch.nn as nn
from typing import Callable
from omegaconf import OmegaConf
from torch.distributed import init_process_group

THOUSAND = 1000
MILLION = 1000000
        
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)  # Recursively create Config objects for nested dictionaries
            setattr(self, key, value)

    @staticmethod
    def from_yaml(file_path):
        # Load the YAML file
        cfg_dict = OmegaConf.load(file_path)
        # Convert OmegaConf object to a regular dictionary
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        # Instantiate the configuration class
        return Config(cfg_dict)
    
    def to_dict(self, keys):
        result = {}
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, Config):
                value = value.to_dict(value.__dict__.keys())  # Recursively convert nested Config objects to dictionaries
            result[key] = value
        return result

class Configs:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)  # Recursively create Config objects for nested dictionaries
            setattr(self, key, value)

    @staticmethod
    def from_yaml(file_path):
        # Load the YAML file
        cfg_dict = OmegaConf.load(file_path)
        # Convert OmegaConf object to a regular dictionary
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        # Instantiate the configuration class
        return Config(cfg_dict)
    
    def to_dict(self, keys):
        result = {}
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, Config):
                value = value.to_dict(value.__dict__.keys())  # Recursively convert nested Config objects to dictionaries
            result[key] = value
        return result

class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            if f!='ckpt_latest.pt' and f!='ckpt_best.pt':
                _, score, it = f.split('_')
                it = it.split('.')[0]
                self.ckpts.append({
                    'score': float(score),
                    'file': f,
                    'iteration': int(it),
                })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None
    
    def save(self, model, args, score, others=None, step=None, save_latest=False, save_best=False):
        
        if save_latest: fname= 'ckpt_latest.pt'
        elif save_best: fname = 'ckpt_best.pt'
        else:
            if step is None:
                fname = 'ckpt_%.6f_.pt' % float(score)
            else:
                fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt

def load_model_state(model, state_dict, multi_gpu_model=False):
    def strip_module(state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    def add_module(state_dict):
        return {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}

    has_module = all(k.startswith("module.") for k in state_dict.keys())

    if multi_gpu_model:
        model = torch.nn.DataParallel(model)
        if not has_module:
            state_dict = add_module(state_dict)
    else:
        if has_module:
            state_dict = strip_module(state_dict)

    model.load_state_dict(state_dict)
    return model
def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix='', start_time=time.localtime()):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', start_time) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


class SubnetFactory:

    class CatCall(nn.Module):

        def __init__(self, layers) -> None:
            super().__init__()
            self.layers = layers

        def forward(self, x, context):
            if context is not None:
                x = torch.cat((x, context), dim=1)
            return self.layers(x)

    def __init__(
                self,
                hidden_features:int,
                context_features:int,
                num_layers:int,
                activation:Callable,
                dropout_probability:float) -> None:
            self.context_features = context_features
            self.hidden_features = hidden_features
            self.num_layers = num_layers
            self.activation = activation
            self.dropout_probability = dropout_probability

    def __call__(self, num_features_in, num_features_out):
        layers = []
        last_features = num_features_in + self.context_features
        for i in range(self.num_layers):
            layers.append(nn.Linear(last_features, self.hidden_features))
            layers.append(self.activation())
            if self.dropout_probability > 0.:
                layers.append(nn.Dropout(self.dropout_probability))
            last_features = self.hidden_features
        layers.append(nn.Linear(last_features, num_features_out))
        return self.CatCall(nn.Sequential(*layers))
    

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))