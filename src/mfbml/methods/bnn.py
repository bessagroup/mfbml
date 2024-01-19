# this script is used to train a Bayesian machine learning model

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
# import Variable
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR


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

    def step(self, closure=None) -> float | None:
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


# class for pre-conditioned Stochastic Gradient Langevin Dynamic optimizer
class pSGLD(Optimizer):
    """ Pre-conditoned Langevin Stochastic Gradient Descent optimizer. It is
     used for bayesian neural networks. It is a variant of SGD and SGLD
     optimizer.

    .. [1] Li, C., Chen, C., Carlson, D., & Carin, L. (2016, February).
    "Preconditioned stochastic gradient Langevin dynamics for deep neural
    networks". In Proceedings of the AAAI conference on artificial
    intelligence (Vol. 30, No. 1).
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

    def step(self, closure=None) -> float | None:
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


# define a neural network class
class LinearNet(nn.Module):
    """ Linear neural network class

    Parameters
    ----------
    nn : nn.Module
        neural network module
    """

    def __init__(self,
                 in_features: int = 1,
                 out_features: int = 1,
                 hidden_features: list = [20, 20, 20],
                 activation: str = "Tanh",
                 prior_mu: float = 0.0,
                 prior_sigma: float = 1.0) -> None:
        super().__init__()

        # define prior distribution for weight and bias
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # define neural architecture
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_features
        self.num_hidden = len(hidden_features)
        self.activation = activation

        # create the nn architecture
        self.net = self._create_nn_architecture()

    def forward(self, x, training=True):
        # define your forward pass here
        x = self.net(x)

        if training:
            # compute the prior loss
            prior_loss = 0
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # prior distribution
                    dist = torch.distributions.Normal(
                        self.prior_mu, self.prior_sigma)
                    # prior loss
                    prior_loss = prior_loss + \
                        dist.log_prob(m.weight).sum() + \
                        dist.log_prob(m.bias).sum()
            return x, -prior_loss
        else:
            return x

    def _create_nn_architecture(self) -> Any:
        """create the nn architecture

        Returns
        -------
        Any
            nn architecture
        """

        # create the first layer
        layers = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layers[0])
        )
        layers.append(self._get_activation())
        for ii in range(1, self.num_hidden):
            layers.append(
                nn.Linear(self.hidden_layers[ii - 1], self.hidden_layers[ii])
            )
            layers.append(self._get_activation())
        # add the last layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.out_features))

        return layers

    def _get_activation(self) -> Any:
        """get activation function according names

        Returns
        -------
        Any
            activation function

        Raises
        ------
        ValueError
            the activation is not implemented in this framework!
        """

        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                "the activation is not implemented in this framework!"
            )


# define a neural network class
class BNNWrapper:
    # define a neural network class
    def __init__(self,
                 in_features: int = 1,
                 out_features: int = 1,
                 hidden_features: list = [20, 20, 20],
                 activation: str = "Tanh",
                 prior_mu: float = 0.0,
                 prior_sigma: float = 1.0,
                 sigma: float = 0.1,
                 lr: float = 0.0001) -> None:
        super().__init__()

        # define the neural network
        self.network = LinearNet(in_features=in_features,
                                 out_features=out_features,
                                 hidden_features=hidden_features,
                                 activation=activation,
                                 prior_mu=prior_mu,
                                 prior_sigma=prior_sigma)
        # define the sigma (standard deviation of the noise)
        self.sigma = sigma
        # define the optimizer (Stochastic Gradient Langevin Dynamic optimizer)
        self.optimizer = pSGLD(self.network.net.parameters(), lr=lr)

    def train(self,
              x: torch.Tensor,
              y: torch.Tensor,
              num_epochs: int,
              sample_freq: int,
              burn_in_epochs: int,
              print_info: bool = True) -> None:
        """train the neural network, to now the batch size is fixed to the
        same as the number of samples. Later the batch size will be implemented
        when the data is large.

        Parameters
        ----------
        x : torch.Tensor
            input data
        y : torch.Tensor
            output data
        num_epochs : int
            number of epochs (including the burn-in epochs and mixing epochs)
        sample_freq : int
            sample frequency (how many epochs to sample a new network)
        burn_in_epochs : int
            number of burn-in epochs
        """
        # learning rate
        scheduler = ExponentialLR(self.optimizer, gamma=0.9999)
        # number of epochs (total epochs)
        self.num_epochs = num_epochs
        # initialize the BNNs for prediction after training
        self.nets = []
        self.log_likelihood = []
        # predefine arrays to store the loss
        self.nll_loss_train = torch.zeros(num_epochs)
        self.prior_loss_train = torch.zeros(num_epochs)
        self.total_loss_train = torch.zeros(num_epochs)
        self.lr_record = []
        # begin training the loop by retain_graph=True
        torch.autograd.set_detect_anomaly(True)
        for ii in range(num_epochs):
            # set gradient of params to zero
            self.optimizer.zero_grad()
            # get prediction from network
            pred, prior_loss = self.network(x, training=True)
            # calculate nll loss
            nll_loss = self._nog_likelihood_function(
                pred=pred,
                real=y,
                sigma=self.sigma,  # type: ignore
                num_dim=self.network.out_features)

            # total loss
            loss = nll_loss + prior_loss
            # back propagation
            loss.backward(retain_graph=True)
            self.optimizer.step()
            scheduler.step()
            # save information
            self.nll_loss_train[ii] = nll_loss.data
            self.prior_loss_train[ii] = prior_loss.data
            self.total_loss_train[ii] = loss.data

            if print_info and ii % 100 == 0:
                self._print_info(iter=ii)

            if ii % sample_freq == 0 and ii >= burn_in_epochs:
                # sample from the posterior
                self.nets.append(copy.deepcopy(self.network))
                self.log_likelihood.append(-nll_loss.data.numpy())
                self.lr_record.append(self.optimizer.param_groups[0]['lr'])

    def predict(self, x: torch.Tensor) -> Any:

        # x = to_variable(var=(x), cuda=False)

        responses = []
        responses_weighted = []
        for ii, net in enumerate(self.nets):
            pred_weighted = net(
                x, training=False).data.numpy()*self.lr_record[ii]
            pred = net(x, training=False).data.numpy()
            responses.append(pred)
            responses_weighted.append(pred_weighted)

        # get the mean and variance of the responses
        pred_mean = (np.array(responses_weighted).sum(axis=0)
                     ).reshape(-1, self.network.out_features) \
            / np.sum(self.lr_record)

        # aleatoric uncertainty
        aleatoric = self.sigma
        # epistemic uncertainty
        epistemic = np.array(responses).std(
            axis=0).reshape(-1, self.network.out_features)
        # total uncertainty
        total_unc = np.sqrt(aleatoric**2 + epistemic**2)

        return pred_mean, epistemic, total_unc, aleatoric

    def _print_info(self, iter: int) -> None:

        noise_p = self.sigma
        print("==================================================")
        print("epoch: %5d/%5d" % (iter, self.num_epochs,))
        print("nll_loss: %2.3f, prior_loss: %2.3f, total: %2.3f" %
              (self.nll_loss_train[iter],
               self.prior_loss_train[iter],
               self.total_loss_train[iter]))
        print("noise: %2.3f" % (noise_p))

    @staticmethod
    def _nog_likelihood_function(pred: torch.Tensor,
                                 real: torch.Tensor,
                                 sigma: torch.Tensor,
                                 num_dim: int) -> torch.Tensor:
        """negative log likelihood function

        Parameters
        ----------
        pred : torch.Tensor
            predicted value from the neural network
        real : torch.Tensor
            real value from the data
        sigma : torch.Tensor
            standard deviation of the noise (hyperparameter)
        num_dim : int
            number of dimensions

        Returns
        -------
        torch.Tensor
            negative log likelihood

        """
        sigma = torch.Tensor([sigma])
        exponent = -0.5*(pred - real)**2/sigma**2
        log_coef = -num_dim*0.5*torch.log(sigma**2)

        neg_lld = - (log_coef + exponent - 0.5 *
                     torch.log(torch.tensor(2*torch.pi))).sum()

        return neg_lld
