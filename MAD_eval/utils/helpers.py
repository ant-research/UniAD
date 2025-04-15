import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# -----------------------------------------------------------------------------#
# -------------------------------- lr_schedule --------------------------------#
# -----------------------------------------------------------------------------#

def get_lr_schedule_with_warmup(optimizer, num_training_steps, last_epoch=-1):
    num_warmup_steps = num_training_steps * 20 / 120
    decay_steps = num_training_steps * 30 / 120

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return max(0., float(current_step) / float(max(1, num_warmup_steps)))
        else:
            return max(0.5 ** ((current_step - num_warmup_steps) // decay_steps), 0.)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -----------------------------------------------------------------------------#
# ---------------------------------- logging ----------------------------------#
# -----------------------------------------------------------------------------#

# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=SummaryWriter, if_exist=False):
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer(log_dir, flush_secs=120, max_queue=10)
        if not if_exist:
            log = logging.getLogger(log_dir)
            if not log.handlers:
                log.setLevel(logging.DEBUG)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
                fh.setFormatter(formatter)
                log.addHandler(fh)
            self.log = log

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def flush(self):
        self._summ_writer.flush()

    def log_info(self, info):
        self.log.info("{}".format(info))