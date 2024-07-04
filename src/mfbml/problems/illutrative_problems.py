import numpy as np
import torch


class mf_Forrester:

    num_dim: int = 1
    num_obj: int = 1
    design_space: np.ndarray = np.array([[0.0, 1.0]])
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]

    def hf(self, x: np.ndarray, noise_std: float = 0.0) -> np.ndarray:

        # copy the input array
        x = np.copy(x)
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)

        return obj

    def lf_1(self, x: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        x = np.copy(x)
        obj = self.hf(x) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)

        return obj

    def lf_2(self, x: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        x = np.copy(x)
        obj = 0.5*self.hf(x) + 10*(x-0.5) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj

    def lf_3(self, x: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        # copy the input array
        x = np.copy(x)
        obj = (5.5 * x - 2.5) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj

    def lf_factor(self,
                  x: np.ndarray,
                  beta_0: float = 0.5,
                  beta_1: float = 5.0,
                  noise_std: float = 0.0) -> np.ndarray:
        # copy the input array
        x = np.copy(x)
        obj = beta_1 * self.hf(x) + 10 * (x - 0.5) - beta_0
        obj = np.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * np.random.randn(*obj.shape)
        return obj


class mf_Forrester_torch:
    """Forrester function 1b
    """

    def __init__(self, noise_std: float) -> None:

        self.noise_std = noise_std

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

    def lf_1(self,
             x: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        x = torch.clone(x)
        obj = self.hf(x) - 5
        obj = torch.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)

        return obj

    def lf_2(self,
             x: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        x = torch.clone(x)
        obj = 0.5*self.hf(x) + 10*(x-0.5) - 5
        obj = torch.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj

    def lf_3(self,
             x: torch.Tensor,
             noise_std: float = 0.0) -> torch.Tensor:
        # copy the input array
        x = torch.clone(x)
        obj = (5.5 * x - 2.5) ** 2 * torch.sin(12 * x - 4)
        obj = torch.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj

    def lf_factor(self,
                  x: torch.Tensor,
                  beta_0: float = 0.5,
                  beta_1: float = 5.0,
                  noise_std: float = 0.0) -> torch.Tensor:
        # copy the input array
        x = torch.clone(x)
        obj = beta_1 * self.hf(x) + 10 * (x - 0.5) - beta_0
        obj = torch.reshape(obj, (x.shape[0], 1))
        # add noise
        obj += noise_std * torch.randn(*obj.shape)
        return obj
