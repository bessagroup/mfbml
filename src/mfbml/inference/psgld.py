# ------------------ Beginning of Reference Python Module ---------------------
""" Module for preconditioned Stochastic Gradient Langevin Dynamics Sampler
using PyTorch.

Classes
-------
pSGLD
    preconditioned Stochastic Gradient Hamiltonian Monte-Carlo Sampler that
    uses a burn-in procedure to adapt its own hyperparameters during the
    initial stages of sampling.
"""
#
#                                                                       Modules
# =============================================================================
# import Variable
import torch
from torch.autograd import Variable
from torch.optim import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class pSGLD(Optimizer):
    """ Pre-conditoned Langevin Stochastic Gradient Descent optimizer. It is
    used for bayesian neural networks. It is a variant of SGD and SGLD
    optimizer.

    """

    def __init__(self,
                 params: dict,
                 lr: float,
                 alpha: float = 0.99,
                 eps: float = 1e-5,
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
        defaults = dict(lr=lr, alpha=alpha, eps=eps)

        # set nestrov
        self.netstrov = nesterov
        # call the super class
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(pSGLD, self).__setstate__(state)
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

                # get the state
                state = self.state[p]

                # initialize the state
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)

                # get the state parameters
                V = state['V']
                # get the parameters
                alpha = group['alpha']
                eps = group['eps']
                lr = group['lr']

                # update the state
                state['step'] += 1

                # update parameters
                # update V
                V.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                # get 1/G use the inplace operation (need division when use it)
                G = V.add(eps).sqrt_()

                # update parameters with 0.5*lr (main update of pSGLD)
                p.data.addcdiv_(d_p, G, value=-lr/2)

                # inject noise to the parameters
                noise_std = torch.sqrt(lr/G)
                noise = Variable(p.data.new(p.size()).normal_())
                p.data.add_(noise_std*noise)

        return loss
