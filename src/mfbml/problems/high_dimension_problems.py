# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the classes for high dimensional problems and it is
implemented in PyTorch.

Classes
-------
MFB1
    A class for the mfb1 function
MengCase1
    A class for Meng's case 1 function
Rosenbrock
    A class for the Rosenbrock function
Paciorek
    A class for the Paciorek function
Meng20D
    A class for Meng's 20D function
Meng4D
    A class for Meng's 4D function

"""

#
#                                                                       Modules
# =============================================================================
# standard library modules
from abc import ABC
from typing import List

# third party modules
import torch

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class MultiFidelityProblem_(ABC):
    """base class for multi-fidelity problems

    Parameters
    ----------
    ABC : class
        abstract base class
    """

    def __call__(self, X: List) -> List:
        """evaluate the function

        Parameters
        ----------
        X: List
            a list of samples, the fist element is the high fidelity samples
            and the second element is the low fidelity samples


        Returns
        -------
        dict
            responses
        """

        # evaluate the problem
        responses = [self.hf(X[0]), self.lf(X[1])]

        return responses

    def hf(self, X: torch.Tensor,
           noise_hf: float = 0.0) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X: torch.Tensor
              high fidelity input x
        noise_hf: float, optional
              noise std, by default None

        Returns
        -------
        torch.Tensor
              outputs
        """

        raise NotImplementedError

    def lf(self, X: torch.Tensor,
           noise_lf: float = 0.0) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X: torch.Tensor
                low fidelity input x
        noise_lf: float, optional
                noise std, by default None

        Returns
        -------
        torch.Tensor
                outputs
        """

        raise NotImplementedError


class MFB1(MultiFidelityProblem_):
    """class for mfb1 function"""

    def __init__(self,
                 num_dim: int,
                 noise_std: float,
                 phi: float) -> None:
        """initialize the function

        Parameters
        ----------
        num_dim : int
            number of dimension
        noise_std : float
            noise standard deviation, assume the noise is Gaussian
        phi : float
            a factor that controls the correlation between the high and low
            fidelity functions
        """
        # get the dimension
        self.num_dim = num_dim
        # set the noise std
        self.noise_std = noise_std
        # set the phi
        self.phi = phi

    def hf(self, X: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        x: torch.Tensor
            high fidelity input x
        noise_hf: float, optional
            noise std, by default None  # type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_hf is None:
            noise_hf = self.noise_std

        obj = (X**2+1-torch.cos(10*torch.pi*X)).sum(dim=1, keepdim=True) + \
            noise_hf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self,
           X: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X: torch.Tensor
            low fidelity input X
        noise_lf: float, optional
            noise std, by default None
        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_lf is None:  # use the default noise
            noise_lf = self.noise_std

        obj = self.hf(X, noise_hf=0.0) + self.error(X) + \
            noise_lf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def error(self, X: torch.Tensor) -> torch.Tensor:
        """error function

        Parameters
        ----------
        X: torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            outputs
        """

        obj = self.a*torch.cos(self.w*X + self.b +
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

    @property
    def theta(self) -> float:
        """theta function

        Returns
        -------
        float
            theta
        """

        return 1 - 0.0001 * self.phi


class MengCase1(MultiFidelityProblem_):
    """Meng's case 1,

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

    def hf(self, X: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X : torch.Tensor
            high fidelity input X
        noise_hf : float, optional
            noise std, by default None#type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """
        # use the default noise
        if noise_hf is None:
            noise_hf = self.noise_std

        obj = torch.sin(8*torch.pi*X)**2*(X -
                                          torch.sqrt(torch.Tensor([2.0])))\
            + noise_hf * torch.randn(X.shape)

        return obj.reshape(-1, 1)

    def lf1(self,
            X: torch.Tensor,
            noise_lf: float = None  # type: ignore
            ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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
        obj = torch.sin(8*torch.pi*X) + \
            noise_lf * torch.randn(X.shape)

        return obj.reshape(-1, 1)

    def lf2(self,
            X: torch.Tensor,
            noise_lf: float = None  # type: ignore
            ) -> torch.Tensor:
        """low fidelity function
        """
        if noise_lf is None:
            noise_lf = self.noise_std
        obj = 1.2*self.hf(X, noise_hf=0.0) - 0.5 +  \
            noise_lf * torch.randn(X.shape)

        return obj.reshape(-1, 1)

    def lf3(self,
            X: torch.Tensor,
            noise_lf: float = None  # type: ignore
            ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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

        obj = torch.sin(16*torch.pi*X)**2 + \
            noise_lf * torch.randn(X.shape)
        return obj.reshape(-1, 1)


class Rosenbrock(MultiFidelityProblem_):

    def __init__(self, num_dim: int,
                 noise_std: float) -> None:

        self.num_dim = num_dim
        self.noise_std = noise_std

    def hf(self, X: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X : torch.Tensor
              high fidelity input X
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
            val = 100 * (X[:, i + 1] - X[:, i] ** 2) ** 2 + (1 - X[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) + \
            noise_hf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, X: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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
            val = 50 * (X[:, i + 1] - X[:, i] ** 2) ** 2 + (-2 - X[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) \
            - 0.5*X.sum(dim=1, keepdim=True) + \
            noise_lf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Paciorek(MultiFidelityProblem_):
    """Paciorek function, design space is [0,1]^d
    """

    def __init__(self, num_dim: int,
                 noise_std: float) -> None:

        self.num_dim = num_dim
        self.noise_std = noise_std

    def hf(self, X: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X : torch.Tensor
              high fidelity input X
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

        obj = torch.sin(1.0/X.prod(dim=1, keepdim=True)) + \
            noise_hf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, X: torch.Tensor,
           noise_lf: float = None,
           A=0.5) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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

        obj = self.hf(X, noise_hf=0.0) + \
            9*A**2*torch.cos(1./X.prod(dim=1, keepdim=True)) +\
            noise_lf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Meng20D(MultiFidelityProblem_):

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

    def hf(self, X: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X : torch.Tensor
              high fidelity input X
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
            val = (2*X[:, i + 1] ** 2 - X[:, i]) ** 2
            list_of_sum.append(val)

        obj = torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) + \
            torch.reshape((X[:, 0]-1.0)**2, (-1, 1)) + noise_hf * \
            torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, X: torch.Tensor,
           noise_lf: float = None) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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
            val = 0.4*(X[:, i + 1]*X[:, i])
            list_of_sum.append(val)

        obj = 0.8*self.hf(X, noise_hf=0.0) - \
            torch.stack(list_of_sum, dim=1).sum(dim=1, keepdim=True) - 50 +\
            noise_lf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)


class Meng4D(MultiFidelityProblem_):

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

    def hf(self, X: torch.Tensor, noise_hf: float = None) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        X : torch.Tensor
              high fidelity input X
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
        obj = 0.5*(0.1*torch.exp(X[:, 0] + X[:, 1]) +
                   X[:, 3]*torch.sin(12*torch.pi*X[:, 2]) + X[:, 2])
        # reshape the obj
        obj = obj.reshape(-1, 1)
        # add noise
        obj = obj + noise_hf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self, X: torch.Tensor, noise_lf: float = None) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        X : torch.Tensor
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

        obj = 1.2*self.hf(X, noise_hf=0.0) - 0.5 + \
            noise_lf * torch.randn(X.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)
