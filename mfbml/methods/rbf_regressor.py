
# =============================================================================
import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from .rbf_kernel import RBF


class RBFSurrogate:

    def __init__(self,
                 design_space: np.ndarray,
                 params_optimize: bool = False,
                 optimizer_restart: int = 0,
                 ) -> None:
        # determine whether to optimize the parameters or not
        self.params_optimize = params_optimize
        # number of restarts for the optimizer
        self.optimizer_restart = optimizer_restart
        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space
        # set kernel
        self.kernel = RBF(theta=np.zeros(self.num_dim))

    def train(self,
              sample_x: np.ndarray,
              sample_y: np.ndarray,
              portion_test: float = 0.2) -> None:
        """Train the surrogate model

        Parameters
        ----------
        sample_x : np.ndarray
            samples with shape=((num_samples, num_dim))
        sample_y : np.ndarray
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
        self.sample_x = sample_x
        self.sample_y = sample_y
        # regularization
        self.sample_x_scaled = self.normalize_input(sample_x=sample_x,
                                                    bounds=self.bounds)
        self.sample_y_scaled = self.normalize_output(sample_y=sample_y)

        if not self.params_optimize:
            self._set_kernel_params(params=np.ones(self.num_dim))
        else:
            self._optimize_kernel_params(params=np.zeros(self.num_dim))

        # get kernel matrix
        self.K = self.kernel.get_kernel_matrix(self.sample_x_scaled,
                                               self.sample_x_scaled)
        # LU decomposition
        self.L = cholesky(self.K)

        # get weights
        self.W = solve(self.L.T, solve(self.L, self.sample_y_scaled))

    def predict(self, x_predict: np.ndarray):
        sample_new = self.normalize_input(x_predict, self.bounds)
        sample_new = np.atleast_2d(sample_new)

        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_x_scaled, sample_new)

        pred = np.dot(self.W.T, knew).reshape(-1, 1)
        # scale back
        pred = pred * self.y_std + self.y_mean

        return pred

    def _set_kernel_params(self, params=None):
        self.kernel.set_params(params=params)

    def _optimize_kernel_params(self, params=None):
        # define objective function
        def mse_loss(params):
            # split samples into two parts
            X_train, X_test, y_train, y_test = train_test_split(
                self.sample_x_scaled.copy(),
                self.sample_y_scaled.copy(),
                test_size=self.portion_test,
                shuffle=True,
                random_state=42)
            self._set_kernel_params(params=params)
            K = self.kernel.get_kernel_matrix(X_train, X_train)
            L = cholesky(K)
            W = solve(L.T, solve(L, y_train))
            knew = self.kernel.get_kernel_matrix(X_train, X_test)
            pred = np.dot(W.T, knew).reshape(-1, 1)
            return np.mean((pred - y_test)**2)
        # optimize parameters using L-BFGS-B algorithm with restart
        n_trials = self.optimizer_restart + 1
        optimum_value = float("inf")
        for _ in range(n_trials):
            # initial point
            x0 = np.random.uniform(
                self.kernel._get_low_bound,
                self.kernel._get_high_bound,
                self.kernel._get_num_para,
            )
            # get the optimum value
            optimum_info = minimize(
                mse_loss,
                x0=x0,
                method="l-bfgs-b",
                bounds=self.kernel._get_bounds_list,
                options={"maxfun": 200},
            )
            # greedy search for the optimum value
            if optimum_info.fun < optimum_value:
                opt_param = optimum_info.x
                optimum_value = optimum_info.fun

        # set parameters
        self._set_kernel_params(params=opt_param)

    def normalize_output(self, sample_y: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_y : np.ndarray
            samples to scale

        Returns
        -------
        np.ndarray
            normalized samples
        """
        self.y_mean = sample_y.mean()
        self.y_std = sample_y.std()

        return (sample_y - self.y_mean) / self.y_std

    @staticmethod
    def normalize_input(sample_x: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_x : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (sample_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


class NoiseRBFSurrogate:

    def __init__(self,
                 design_space: np.ndarray,
                 noise_std: float = 0.1) -> None:
        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space

        # noise level
        self.noise_std = noise_std
        self.kernel = RBF(theta=np.zeros(self.num_dim))
        self._set_kernel_params(params=np.ones(self.num_dim))

    def _set_kernel_params(self, params=None):
        self.kernel.set_params(params=params)

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:

        # get samples
        self.sample_x = sample_x
        self.sample_y = sample_y
        # regularization
        self.sample_x_scaled = self.normalize_input(sample_x=sample_x,
                                                    bounds=self.bounds)
        self.sample_y_scaled = self.normalize_output(sample_y=sample_y)
        # get kernel matrix
        self.K = self.kernel.get_kernel_matrix(self.sample_x_scaled,
                                               self.sample_x_scaled) + \
            (self.noise_std/self.y_std)**2 * np.eye(self.sample_x.shape[0])
        # LU decomposition
        self.L = cholesky(self.K)

        # get weights
        self.W = solve(self.L.T, solve(self.L, self.sample_y_scaled))

    def predict(self, x_predict: np.ndarray):
        sample_new = self.normalize_input(x_predict, self.bounds)
        sample_new = np.atleast_2d(sample_new)

        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_x_scaled, sample_new)

        pred = np.dot(self.W.T, knew).reshape(-1, 1)
        # scale back
        pred = pred * self.y_std + self.y_mean
        return pred

    def normalize_output(self, sample_y) -> np.ndarray:
        """Normalize samples to normal distribution

        Parameters
        ----------
        sample_y : np.ndarray
            samples to scale

        Returns
        -------
        np.ndarray
            normalized samples
        """
        self.y_mean = sample_y.mean()
        self.y_std = sample_y.std()

        return (sample_y - self.y_mean) / self.y_std

    @staticmethod
    def normalize_input(sample_x: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_x : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (sample_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
