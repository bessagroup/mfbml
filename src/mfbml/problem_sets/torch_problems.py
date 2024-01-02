# define problems using pytorch
# =========================================================================== #


from typing import Any

import torch


class MengCase1:
    """Meng's case 1, 

    reference:
    ---------
    [1] Meng, Xuhui / Babaee, Hessam / Karniadakis, George Em 
    Multi-fidelity Bayesian neural networks: Algorithms and applications 
    2021 

    """

    def __init__(self, noise_std: float = 0.0):
        """Meng's case 1

        Parameters
        ----------
        noise_std : float, optional
            noise standard deviation, by default 0.0
        """
        # noise standard deviation
        self.noise_std = noise_std

    def __call__(self, samples: dict) -> dict:
        """evaluate the problem

        Parameters
        ----------
        samples : dict
            samples

        Returns
        -------
        dict
            responses
        """

        # get samples
        hf_samples = samples["hf"]
        lf_samples = samples["lf"]

        # evaluate the problem
        responses = {"hf": self.hf(hf_samples),
                     "lf": self.lf(lf_samples)}

        return responses

    def hf(self, x: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        x : torch.Tensor
            high fidelity input x
        noise_hf : float, optional
            noise std, by default None#type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_hf is None:  # use the default noise
            noise_hf = self.noise_std

        obj = torch.sin(8*torch.pi*x)**2*(x - torch.sqrt(torch.Tensor([2.0]))) + \
            noise_hf * torch.randn(x.shape)

        return obj.reshape(-1, 1)

    def lf(self,
           x: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        noise_lf : float, optional
            noise standard deviation, by default None

        Returns
        -------
        torch.Tensor
            output tensor
        """

        if noise_lf is None:  # use the default noise
            noise_lf = self.noise_std
        obj = torch.sin(8*torch.pi*x) + \
            noise_lf * torch.randn(x.shape)

        return obj.reshape(-1, 1)


class Forrester1b:
    """Forrester function 1b
    """

    def __init__(self, noise_std: float) -> None:

        self.noise_std = noise_std

    def __call__(self, samples: dict) -> dict:
        """evaluate the problem

        Parameters
        ----------
        samples : dict
            samples

        Returns
        -------
        dict
            responses
        """

        # get samples
        hf_samples = samples["hf"]
        lf_samples = samples["lf"]

        # evaluate the problem
        responses = {"hf": self.hf(hf_samples),
                     "lf": self.lf(lf_samples)}

        return responses

    def hf(self, x: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        x : torch.Tensor
            high fidelity input x
        noise_hf : float, optional
            noise std, by default None 

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_hf is None:
            noise_hf = self.noise_std

        obj = (6*x - 2)**2 * torch.sin(12*x - 4) + \
            noise_hf * torch.randn(x.shape)

        return obj.reshape(-1, 1)

    def lf(self, x: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        noise_lf : float, optional
            noise standard deviation, by default None

        Returns
        -------
        torch.Tensor
            output tensor
        """

        if noise_lf is None:
            noise_lf = self.noise_std

        obj = 0.5 * self.hf(x, noise_hf=0.0) + \
            10 * (x - 0.5) - 5 + \
            noise_lf * torch.randn(x.shape)

        return obj.reshape(-1, 1)


# # test case
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # create problem
#     problem = Forrester1b(noise_std=0.0)

#     # create data
#     x = torch.linspace(0, 1, 1000).reshape(-1, 1)
#     y_hf = problem.hf(x)
#     y_lf = problem.lf(x)

#     # plot
#     plt.plot(x, y_hf, label='hf')
#     plt.plot(x, y_lf, label='lf')
#     plt.legend()
#     plt.show()
