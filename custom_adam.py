# Copyright 2019-2020 Stanislav Pidhorskyi
# lr_equalization_coef was added for LREQ

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# https://github.com/pytorch/pytorch/blob/master/LICENSE


import math
import torch
from torch.optim.optimizer import Optimizer


class LREQAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.0, 0.99), eps=1e-8,
                 weight_decay=0):
        assert betas[0] == 0, "LREQAdam does not use first moment of gradient at all"
        beta_2 = betas[1]
        defaults = dict(lr=lr, beta_2=beta_2, eps=eps,
                        weight_decay=weight_decay)
        super(LREQAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LREQAdam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg_sq = state['exp_avg_sq']
                beta_2 = group['beta_2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data / p.coef)

                # Decay the second moment running average coefficient
                exp_avg_sq.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction2 = 1 - beta_2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2)

                if hasattr(p, 'lr_equalization_coef'):
                    step_size *= p.lr_equalization_coef

                p.data.addcdiv_(grad, denom, value=-step_size)

        return loss
