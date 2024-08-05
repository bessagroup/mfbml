# ------------------ Beginning of Reference Python Module ---------------------
""" Module for Stochastic Gradient Hamiltonian MC Sampler using PyTorch.

Classes
-------
SGHMC
    Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
    procedure to adapt its own hyperparameters during the initial stages
    of sampling. Code is adapted from the original implementation in
    `torch.optim.SGD`.
"""
#
#                                                                       Modules
# =============================================================================
# third party modules
import numpy as np
import torch
from torch.optim import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================
class SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.
    """
    name = "SGHMC"

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 mdecay: float = 0.01,
                 wd: float = 0.00002,
                 scale_grad: float = 1.) -> None:
        """ Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr, scale_grad=scale_grad,
            mdecay=mdecay,
            wd=wd
        )
        super().__init__(params, defaults)

    def step(self, closure=None) -> float | None:
        """ Performs a single optimization step.

        Parameters
        ----------
        closure : optional
            default torch optimizer variable, by default None

        Returns
        -------
        float | None
            loss value
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(parameter.shape) == 1 and parameter.shape[0] == 1:
                    # optimize aleatoric noise without adding nosie
                    parameter.data.add_(
                        parameter.grad.data, alpha=-group["lr"])
                else:
                    # use SGHMC to optimize parameters of NN
                    if len(state) == 0:
                        state["iteration"] = 0
                        state["momentum"] = torch.randn(
                            parameter.size(), dtype=parameter.dtype)

                    state["iteration"] += 1

                    mdecay, lr = group["mdecay"], group["lr"]
                    scale_grad = group["scale_grad"]

                    momentum = state["momentum"]
                    gradient = parameter.grad.data * scale_grad
                    # set beta to be zero in this case
                    sigma = torch.sqrt(torch.from_numpy(
                        np.array(2 * lr * mdecay, dtype=type(lr))))
                    sample_t = torch.normal(mean=torch.zeros_like(
                        gradient), std=torch.ones_like(gradient) * sigma)

                    # update the momentum and parameters
                    parameter.data.add_(lr * mdecay * momentum)
                    momentum.add_(
                        -lr * gradient - mdecay * lr * momentum + sample_t)
        return loss
