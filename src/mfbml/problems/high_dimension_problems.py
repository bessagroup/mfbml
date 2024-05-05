# this script is used for implementing mfb test suites from Handing Wang's 2018
# paper


from typing import Any

import torch


class MFB1:
    """class for mfb1 function"""

    def __init__(self,
                 num_dim: int,
                 noise_std: float,
                 phi: float) -> None:
        """constructor"""
        # get the dimension
        self.num_dim = num_dim
        # set the noise std
        self.noise_std = noise_std
        # set the phi
        self.phi = phi

    def __call__(self, samples: dict) -> dict:
        """evaluate the function

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

        obj = (x**2+1-torch.cos(10*torch.pi*x)).sum(dim=1, keepdim=True) + \
            noise_hf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self,
           x: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        x : torch.Tensor
            low fidelity input x
        noise_lf : float, optional
            noise std, by default None#type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_lf is None:  # use the default noise
            noise_lf = self.noise_std

        obj = self.hf(x, noise_hf=0.0) + self.error(x) + \
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """error function

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            outputs
        """

        obj = self.a*torch.cos(self.w*x + self.b +
                               torch.pi).sum(dim=1, keepdim=True)

        return obj.reshape(-1, 1)

    @property
    def a(self) -> float:
        """a function

        Returns
        -------
        float
            a
        """

        return self.theta

    @property
    def b(self) -> float:
        """b function

        Returns
        -------
        float
            b
        """

        return 0.5*torch.pi*self.theta

    @property
    def w(self) -> float:
        """w function

        Returns
        -------
        float
            w
        """

        return 10.0*torch.pi*self.theta

    @ property
    def theta(self) -> float:
        """theta function

        Returns
        -------
        float
            theta
        """

        return 1 - 0.0001 * self.phi


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


class Rosenbrock:

    def __init__(self, num_dim: int,
                 noise_std: float) -> None:

        self.num_dim = num_dim
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

        # use a for loop to compute the sum
        list_of_sum = []
        for i in range(self.num_dim - 1):
            val = 100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) + \
            noise_hf * torch.randn(x.shape[0]).reshape(-1, 1)

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

        # use a for loop to compute the sum
        list_of_sum = []
        for i in range(self.num_dim - 1):
            val = 50 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (-2 - x[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) \
            - 0.5*x.sum(dim=1, keepdim=True) + \
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Paciorek:
    """Paciorek function, design space is [0,1]^d
    """

    def __init__(self, num_dim: int,
                 noise_std: float) -> None:

        self.num_dim = num_dim
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

    def hf(self, x: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
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

        # use a for loop to compute the sum

        obj = torch.sin(1.0/x.prod(dim=1, keepdim=True)) + \
            noise_hf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, x: torch.Tensor,
           noise_lf: float = None,
           A=0.5) -> torch.Tensor:
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

        # use a for loop to compute the sum

        obj = self.hf(x, noise_hf=0.0) + \
            9*A**2*torch.cos(1./x.prod(dim=1, keepdim=True)) +\
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Meng20D:

    def __init__(self,
                 num_dim: int = 20,
                 noise_std: float = 0.0) -> None:
        """get required information

        Parameters
        ----------
        num_dim : int, optional
            number of dimension, by default 20
        noise_std : float, optional
            noise level, by default 0.0
        """

        self.noise_std = noise_std
        self.num_dim = num_dim

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

    def hf(self, x: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
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

        # use a for loop to compute the sum
        list_of_sum = []
        for i in range(self.num_dim - 1):
            val = (2*x[:, i + 1] ** 2 - x[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) + \
            torch.reshape((x[:, 0]-1.0)**2, (-1, 1)) + noise_hf * \
            torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, x: torch.Tensor,
           noise_lf: float = None) -> torch.Tensor:
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

        list_of_sum = []
        for i in range(self.num_dim - 1):
            val = 0.4*(x[:, i + 1]*x[:, i])
            list_of_sum.append(val)

        obj = 0.8*self.hf(x, noise_hf=0.0) - \
            torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) - 50 +\
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Meng4D:

    def __init__(self,
                 noise_std: float = 0.0) -> None:
        """get required information

        Parameters
        ----------
        noise_std : float, optional
            noise level, by default 0.0
        """

        self.noise_std = noise_std
        self.num_dim = 4

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

    def hf(self, x: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
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

        # use a for loop to compute the sum
        obj = 0.5*(0.1*torch.exp(x[:, 0] + x[:, 1]) +
                   x[:, 3]*torch.sin(12*torch.pi*x[:, 2]) + x[:, 2])
        # reshape the obj
        obj = obj.reshape(-1, 1)
        # add noise
        obj = obj + noise_hf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, x: torch.Tensor, noise_lf: float = None) -> torch.Tensor:
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

        obj = 1.2*self.hf(x, noise_hf=0.0) - 0.5 + \
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)
