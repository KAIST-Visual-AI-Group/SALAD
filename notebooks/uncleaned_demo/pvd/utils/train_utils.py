import torch
import math


class PolyDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, init_lr, power=0.99, lr_end=1e-7, last_epoch=-1):
        def lr_lambda(step):
            lr = max(power**step, lr_end / init_lr)
            return lr

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def get_dropout_mask(shape, dropout: float, device):
    if dropout == 1:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    elif dropout == 0:
        return torch.ones_like(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) > dropout
