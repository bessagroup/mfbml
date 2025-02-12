# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the classes for the multi-fidelity Forrester function,
it is been implemented in both numpy and pytorch.

Classes
-------
mf_Forrester
    Forrester function in numpy
mf_Forrester_torch
    Forrester function in pytorch

"""

#
#                                                                       Modules
# =============================================================================
# third party modules
import numpy as np
import torch

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class mf_Forrester:
    """Forrester function implementation in numpy
    """

    num_dim: int = 1
    num_obj: int = 1
    design_space: np.ndarray = np.array([[0.0, 1.0]])
    input_domain: np.ndarray = np.array([[0.0, 1.0]])
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]

    def hf(self, X: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """high fidelity function

        Parameters
        ----------
        X : np.ndarray
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        np.ndarray
            outputs of the high fidelity function
        """

        # copy the input array
        X = np.copy(X)
        obj = (6 * X - 2) ** 2 * np.sin(12 * X - 4)
        obj = np.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)

        return obj

    def lf_1(self, X: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """low fidelity function 1

        Parameters
        ----------
        X : np.ndarray
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        np.ndarray
            outputs of the low fidelity function 1
        """
        X = np.copy(X)
        obj = self.hf(X) - 5
        obj = np.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)

        return obj

    def lf_2(self, X: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """low fidelity function 2

        Parameters
        ----------
        X : np.ndarray
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        np.ndarray
            outputs of the low fidelity function 2
        """
        X = np.copy(X)
        obj = 0.5*self.hf(X) + 10*(X-0.5) - 5
        obj = np.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj

    def lf_3(self, X: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """low fidelity function 3

        Parameters
        ----------
        X : np.ndarray
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        np.ndarray
            outputs of the low fidelity function 3
        """
        # copy the input array
        X = np.copy(X)
        obj = (5.5 * X - 2.5) ** 2 * np.sin(12 * X - 4)
        obj = np.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj

    def lf_factor(self,
                  X: np.ndarray,
                  beta_0: float = 0.5,
                  beta_1: float = 5.0,
                  noise_std: float = 0.0) -> np.ndarray:
        """low fidelity function with factor, it can be used to generate
        multi-fidelity data with different fidelity levels

        Parameters
        ----------
        X : np.ndarray
            input array with shape (n, d)
        beta_0 : float, optional
            a factor control how much the HF been used, by default 0.5
        beta_1 : float, optional
            bias factor, by default 5.0
        noise_std : float, optional
            standard deviation, by default 0.0

        Returns
        -------
        np.ndarray
            outputs of the low fidelity function with factor
        """
        # copy the input array
        X = np.copy(X)
        obj = beta_1 * self.hf(X) + 10 * (X - 0.5) - beta_0
        obj = np.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj


class mf_Forrester_torch:
    """Forrester function implementation in pytorch
    """

    def __init__(self, noise_std: float) -> None:
        """initialize the class

        Parameters
        ----------
        noise_std : float
            noise standard deviation
        """

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

        obj = (6*X - 2)**2 * torch.sin(12*X - 4) + \
            noise_hf * torch.randn(X.shape)

        return obj.reshape(-1, 1)

    def lf_1(self,
             X: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        """low fidelity function 1

        Parameters
        ----------
        X : torch.Tensor
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        torch.Tensor
            outputs of the low fidelity function 1
        """
        X = torch.clone(X)
        obj = self.hf(X) - 5
        obj = torch.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)

        return obj

    def lf_2(self,
             X: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        """low fidelity function 2

        Parameters
        ----------
        X : torch.Tensor
            input array with shape (n, d)
        noise_std : float, optional
            standard deviation, by default 0.0

        Returns
        -------
        torch.Tensor
            outputs of the low fidelity function 2
        """
        X = torch.clone(X)
        obj = 0.5*self.hf(X) + 10*(X-0.5) - 5
        obj = torch.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj

    def lf_3(self,
             X: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        """low fidelity function 3

        Parameters
        ----------
        X : torch.Tensor
            input array with shape (n, d)
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        torch.Tensor
            outputs of the low fidelity function 3
        """
        # copy the input array
        X = torch.clone(X)
        obj = (5.5 * X - 2.5) ** 2 * torch.sin(12 * X - 4)
        obj = torch.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj

    def lf_factor(self,
                  X: torch.Tensor,
                  beta_0: float = 0.5,
                  beta_1: float = 5.0,
                  noise_std: float = 0.0) -> torch.Tensor:
        """low fidelity function with factor, it can be used to generate
        multi-fidelity data with different fidelity levels

        Parameters
        ----------
        X : torch.Tensor
            input array with shape (n, d)
        beta_0 : float, optional
            a factor control how much the HF been used, by default 0.5
        beta_1 : float, optional
            bias factor, by default 5.0
        noise_std : float, optional
            noise standard deviation, by default 0.0

        Returns
        -------
        torch.Tensor
            outputs of the low fidelity function with factor
        """
        # copy the input array
        X = torch.clone(X)
        obj = beta_1 * self.hf(X) + 10 * (X - 0.5) - beta_0
        obj = torch.reshape(obj, (X.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj
