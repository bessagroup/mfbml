# ------------------ Beginning of Reference Python Module ---------------------
""" Module for Kernel Ridge Regression (KRR) method

This module contains the classes and functions for training kernel ridge
regression (KRR) models using numpy and scipy libraries.

Classes
-------
RBFKernelRegression
    A class for training the RBF kernel regression model

"""

#                                                                       Modules
# =============================================================================
from typing import List

# third party modules
import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# local modules
from .kernels import RBF

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================

class KernelRidgeRegression:
    """RBF kernel regression, which is used to train the low fidelity model
    for the KRR-LR-GPR method.
    """

    def __init__(self,
                 design_space: np.ndarray,
                 params_optimize: bool = True,
                 optimizer_restart: int = 0,
                 noise_data: bool = False,
                 noise_std: float = 0.1,
                 kernel: RBF = None,
                 seed: int = 42
                 ) -> None:
        """Initialize the RBF kernel ridge regression model

        Parameters
        ----------
        design_space : np.ndarray
            design space with shape=((num_dim, 2))
        params_optimize : bool, optional
            optimize the hyper-parameter of krr or not, by default True
        optimizer_restart : int, optional
            restart the optimizer (l-bfgs-b), by default 0
        noise_data : bool, optional
            Does the data noisy or not, by default False
        noise_std : float, optional
            standard deviation of noise, by default 0.1
        seed : int, optional
            seed for replication, by default 42
        """

        # set random seed
        self.seed = seed
        np.random.seed(seed)
        # determine whether to optimize the parameters or not
        self.params_optimize = params_optimize
        # number of restarts for the optimizer
        self.optimizer_restart = optimizer_restart
        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space
        # noise information
        self.noise_data = noise_data
        self.noise_std = noise_std

        # set kernel
        self.kernel = kernel if kernel else RBF(theta=np.ones(self.num_dim))

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              portion_test: float = 0.2) -> None:
        """Train the KRR model

        Parameters
        ----------
        X : np.ndarray
            samples with shape=((num_samples, num_dim))
        Y : np.ndarray
            responses with shape=((num_samples, 1))
        portion_test : float, optional
            portion of samples to optimize the parameters, by default 0.8

        Returns
        -------
        None
        """
        # portion of samples to optimize the parameters
        self.portion_test = portion_test
        # get samples
        self.sample_x = X
        self.sample_y = Y
        # regularization
        self.sample_x_scaled = self.normalize_input(X=X,bounds=self.bounds)
        self.sample_y_scaled = self.normalize_output(Y=Y)

        if not self.params_optimize:
            self._set_kernel_params(params=np.ones(self.num_dim))
        else:
            self._optimize_kernel_params()

        # get kernel matrix
        self.K = self._calculate_training_kernel_matrix(
            scaled_x=self.sample_x_scaled,
            scaled_noise_std=self.noise_std/self.y_std)

        # LU decomposition
        self.L = cholesky(self.K)

        # get weights
        self.W = solve(self.L.T, solve(self.L, self.sample_y_scaled))

    def predict(self, X: np.ndarray)-> np.ndarray:
        """Predict the responses of samples

        Parameters
        ----------
        X : np.ndarray
            unscaled samples with shape=((num_samples, num_dim))

        Returns
        -------
        np.ndarray
            predicted responses with shape=((num_samples, 1))
        """
        sample_new = self.normalize_input(X, self.bounds)
        sample_new = np.atleast_2d(sample_new)

        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_x_scaled, sample_new)

        pred = np.dot(self.W.T, knew).reshape(-1, 1)
        # scale back
        pred = pred * self.y_std + self.y_mean

        return pred

    def _set_kernel_params(self, params: np.ndarray = None) -> None:
        """Set the parameters of the kernel

        Parameters
        ----------
        params : np.ndarray, optional
            parameters for the kernel, by default None
        """

        if self.noise_data and self.params_optimize:
            # we only optimize the noise level when noise_data is True
            # and params_optimize is True; otherwise, we use the default
            # noise level (0.1 or the value set by users)
            self.kernel.set_params(params=params[:-1])
            self.noise_std = params[-1]*self.y_std
        else:
            # other cases
            self.kernel.set_params(params=params)

    def _optimize_kernel_params(self) -> None:
        """Optimize the parameters of the kernel
        """
        # define objective function
        def mse_loss(params):
            # split samples into two parts
            X_train, X_test, y_train, y_test = train_test_split(
                self.sample_x_scaled.copy(),
                self.sample_y_scaled.copy(),
                test_size=self.portion_test,
                shuffle=True,
                random_state=self.seed)

            # set parameters
            self._set_kernel_params(params=params)
            #
            K = self._calculate_training_kernel_matrix(
                scaled_x=X_train,
                scaled_noise_std=self.noise_std/self.y_std)
            # LU decomposition of K
            L = cholesky(K)
            # get weights
            W = solve(L.T, solve(L, y_train))
            # get the kernel matrix for predicted samples(scaled samples)
            knew = self.kernel.get_kernel_matrix(X_train, X_test)
            # get predicted values (at the scaled level)
            pred = np.dot(W.T, knew).reshape(-1, 1)

            return np.mean((y_test - pred)**2)

        # optimize parameters using L-BFGS-B algorithm with restart
        n_trials = self.optimizer_restart + 1
        optimum_value = float("inf")
        for _ in range(n_trials):
            bounds = self._bound_definition_for_optimization()
            # initial point
            x0 = np.random.uniform(
                bounds[:, 0].tolist(),
                bounds[:, 1].tolist(),
                bounds.shape[0],
            )
            # get the optimum value
            optimum_info = minimize(
                mse_loss,
                x0=x0,
                method="l-bfgs-b",
                bounds=bounds.tolist(),
            )
            # greedy search for the optimum value
            if optimum_info.fun < optimum_value:
                opt_param = optimum_info.x
                optimum_value = optimum_info.fun
        # set parameters
        self._set_kernel_params(params=opt_param)

    def _bound_definition_for_optimization(
            self, noise_bound: List = [10**-3, 10]) -> np.ndarray:
        """Define the bounds for optimization

        Parameters
        ----------
        noise_bound : List, optional
            noise standard deviation bound, by default [10**-3, 10]

        Returns
        -------
        np.ndarray
            bounds for optimization
        """

        if not self.noise_data or not self.params_optimize:
            # bounds for parameters
            bounds = np.zeros((self.num_dim, 2))
            bounds[:, 0] = self.kernel._get_low_bound
            bounds[:, 1] = self.kernel._get_high_bound

        else:
            # bounds for parameters
            bounds = np.zeros((self.num_dim+1, 2))
            bounds[:-1, 0] = self.kernel._get_low_bound
            bounds[:-1, 1] = self.kernel._get_high_bound
            # bounds for noise level
            bounds[-1, 0] = noise_bound[0]
            bounds[-1, 1] = noise_bound[1]

        return bounds

    def _calculate_training_kernel_matrix(self,
                                          scaled_x: np.ndarray,
                                          scaled_noise_std: float) -> np.ndarray:
        """Calculate the kernel matrix for training samples

        Parameters
        ----------
        scaled_x : np.ndarray
            scaled samples
        scaled_noise_std : float
            scaled noise level

        Returns
        -------
        np.ndarray
            kernel matrix for training samples
        """

        if self.noise_data:
            return self.kernel.get_kernel_matrix(scaled_x, scaled_x) + \
                scaled_noise_std**2 * np.eye(scaled_x.shape[0])
        else:
            return self.kernel.get_kernel_matrix(scaled_x, scaled_x)

    def normalize_output(self, Y: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        Y : np.ndarray
            samples to scale

        Returns
        -------
        np.ndarray
            normalized samples
        """
        self.y_mean = Y.mean()
        self.y_std = Y.std()

        return (Y - self.y_mean) / self.y_std

    @staticmethod
    def normalize_input(X: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        X : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
