from typing import Any, Dict

import torch
import torch.nn as nn
# import Variable
from torch.autograd import Variable
from torch.optim import Optimizer

# class for Langevin_SGD optimizer


class SGLD(Optimizer):
    """Langevin Stochastic Gradient Descent optimizer. It is used for 
    bayesian neural networks. It is a variant of SGD optimizer.
    """

    def __init__(self,
                 params: dict,
                 lr: float,
                 nesterov: bool = False) -> None:
        """initialization of Langevin SGD

        Parameters
        ----------
        params : dict
            a dict contains all parameters
        lr : float
            learning rate
        weight_decay : float, optional
            weight decay, by default 0.0
        nesterov : bool, optional
            nesterov, by default False

        """
        if lr < 0.0:
            # learning rate should be positive
            raise ValueError("Invalid learning rate: {}".format(lr))

        # set the default values
        defaults = dict(lr=lr)

        # set nestrov
        self.netstrov = nesterov
        # call the super class
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGLD, self).__setstate__(state)
        # change default state values for param groups
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None) -> float:
        """Performs a single optimization step."""

        loss = None
        # first case
        if closure is not None:
            loss = closure()
        # loop over the parameters
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if len(p.shape) == 1 and p.shape[0] == 1:
                    # for aleatoric noise, no Langevin dynamics is involved.
                    p.data.add_(d_p, alpha=-group['lr'])
                else:
                    # add unit noise to the weights and bias
                    unit_noise = Variable(p.data.new(p.size()).normal_())
                    # Langevin dynamics update
                    p.data.add_(0.5*d_p, alpha=-group['lr'])
                    p.data.add_(unit_noise, alpha=group['lr']**0.5)

        return loss
