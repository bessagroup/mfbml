import numpy as np


class mf_Forrester:

    num_dim: int = 1
    num_obj: int = 1
    design_space: np.ndarray = np.array([[0.0, 1.0]])
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]

    def hf(self, x: np.ndarray) -> np.ndarray:

        # copy the input array
        x = np.copy(x)
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    def lf_1(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        obj = self.hf(x) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj

    def lf_2(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        obj = 0.5*self.hf(x) + 10*(x-0.5) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj

    def lf_3(self, x: np.ndarray) -> np.ndarray:
        # copy the input array
        x = np.copy(x)
        obj = (5.5 * x - 2.5) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj

    def lf_factor(self,
                  x: np.ndarray,
                  beta_0: float = 0.5,
                  beta_1: float = 5.0) -> np.ndarray:
        # copy the input array
        x = np.copy(x)
        obj = beta_1 * self.hf(x) + 10 * (x - 0.5) - beta_0
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj
