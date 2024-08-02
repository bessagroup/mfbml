# ------------------ Beginning of Reference Python Module ---------------------
""" Module for  kernels used in the kernel ridge regression + Linear transfer-
learning + Gaussian process regression model.



Classes
-------
KernelRidgeLinearGaussianProcess
    A class for the proposed KRR-LR-GPR model.

"""

#                                                                       Modules
# =============================================================================
# standard library modules
import time
from abc import ABC
from typing import Any, List, Tuple

# third party modules
import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

# local functions
from .kernel_ridge_regression import KernelRidgeRegression
from .kernels import RBF

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class KernelRidgeLinearGaussianProcess(ABC):
    """A class for the kernel ridge regression model + linear transfer model
     Gaussian process regression model.

    Parameters
    ----------
    ABC : ABC
        abstract base class
    """

    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 0,
        kernel: RBF = None,
        noise_prior: float = None,
        lf_model: KernelRidgeRegression = None,
        lf_poly_order: str = "linear",
        seed: int = 42,
    ) -> None:
        """Initialize the model of KRR + linear transfer model + GPR

        Parameters
        ----------
        design_space : np.ndarray
            design space of this problem
        optimizer : Any, optional
            third party optimizer, by default None
        optimizer_restart : int, optional
            restart the optimizer, by default 0
        kernel : RBF, optional
            kernel for Gaussian process regression, by default None
        noise_prior : float, optional
            noise standard deviation, by default None
        lf_model : KernelRidgeRegression, optional
            low-fidelity model, by default None
        lf_poly_order : str, optional
            polynomial order of , by default "linear"
        seed : int, optional
            numpy seed for replication, by default 42
        """

        np.random.seed(seed)
        self.bounds = design_space
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]

        # get the noise level
        self.noise = noise_prior
        # define kernel
        self.kernel = kernel if kernel else RBF(theta=np.ones(self.num_dim))

        # get lf polynomial order
        self.lf_poly_order = lf_poly_order

        # define the lf model
        self.lf_model = lf_model if lf_model else \
            KernelRidgeRegression(design_space=self.bounds,
                                  params_optimize=True,
                                  noise_data=True,
                                  optimizer_restart=optimizer_restart,
                                  seed=seed)

    def train(self, X: List, Y: List) -> None:
        """Train the hierarchical gaussian process model

        Parameters
        ----------
        samples : dict
            dict with two keys, 'hf' contains np.ndarray of
            high-fidelity sample points and 'lf' contains
            low-fidelity
        responses : dict
            dict with two keys, 'hf' contains high-fidelity
            responses and 'lf' contains low-fidelity ones
        """

        # get samples and normalize them
        self.sample_xh = X[0]
        self.sample_xl = X[1]
        self.sample_xh_scaled = self.normalize_input(self.sample_xh)
        self.sample_xl_scaled = self.normalize_input(self.sample_xl)

        # get responses and normalize them
        self.sample_yh = Y[0]
        self.sample_yl = Y[1]
        self.sample_yh_scaled = self.normalize_hf_output(self.sample_yh)
        self.sample_yl_scaled = (self.sample_yl - self.yh_mean) / self.yh_std
        # rbf surrogate model would normalize the inputs directly
        start_time = time.time()
        self.lf_model.train(X[1], Y[1])
        lf_train_time = time.time()
        # prediction of low-fidelity at high-fidelity locations
        self.f = self._basis_function(self.sample_xh,
                                      poly_order=self.lf_poly_order)
        # optimize the hyper parameters of kernel
        self._optimize_parameters()
        # update parameters
        self._update_parameters()
        end_time = time.time()
        self.lf_training_time = lf_train_time - start_time
        self.hf_training_time = end_time - lf_train_time

    def predict(self,
                X: np.ndarray,
                return_std: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:
        """get the prediction of the kernel ridge regression model + linear
        transfer model + Gaussian process regression model

        Parameters
        ----------
        X : np.ndarray
            unknown samples with shape (n_samples, n_features)
        return_std : bool, optional
            get the predictive uncertainty or not, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of predicted mean and standard deviation
        """
        # normalize the input
        sample_new = self.normalize_input(X)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_xh_scaled, sample_new)
        # calculate the predicted mean
        f = self._basis_function(X, poly_order=self.lf_poly_order)
        # get the mean
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        fmean = (fmean * self.yh_std + self.yh_mean).reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            R = f.T - np.dot(self.f.T, delta)
            # epistemic uncertainty calculation
            mse = self.sigma2 * \
                (1 - np.diag(np.dot(knew.T, delta)) +
                    np.diag(R.T.dot(solve(self.ld.T, solve(self.ld, R))))
                 )
            std = np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)
            # epistemic uncertainty scale back
            self.epistemic = std*self.yh_std

            # total uncertainty
            total_unc = np.sqrt(self.epistemic**2 + self.noise**2)
            return fmean, total_unc

    def _optimize_parameters(self) -> None:
        """optimize the parameter of the model (parameters of the GPR model)
        """

        if self.noise is None:
            # noise value needs to be optimized
            lower_bound_theta = self.kernel._get_low_bound
            upper_bound_theta = self.kernel._get_high_bound
            # set up the bounds for noise sigma
            lower_bound_sigma = 1e-5
            upper_bound_sigma = 10.0
            # set up the bounds for the hyper-parameters
            lower_bound = np.hstack((lower_bound_theta, lower_bound_sigma))
            upper_bound = np.hstack((upper_bound_theta, upper_bound_sigma))
            # bounds for the hyper-parameters
            hyper_bounds = np.vstack((lower_bound, upper_bound)).T
            # number of hyper-parameters
            num_hyper = self.kernel._get_num_para + 1
        else:
            lower_bound = self.kernel._get_low_bound
            upper_bound = self.kernel._get_high_bound
            # bounds for the hyper-parameters
            hyper_bounds = np.vstack((lower_bound, upper_bound)).T
            # number of hyper-parameters
            num_hyper = self.kernel._get_num_para

        if self.optimizer is None:
            n_trials = self.optimizer_restart + 1
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    lower_bound,
                    upper_bound,
                    num_hyper,
                )
                optRes = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=hyper_bounds,
                )
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=num_hyper,
                design_space=hyper_bounds,
            )
            opt_param = optRes["best_x"]
        self.opt_param = opt_param

    def _logLikelihood(self, params: np.ndarray) -> np.ndarray:
        """calculate the ln-concentrated likelihood of the Gaussian process
        model

        Parameters
        ----------
        params : np.ndarray
            a set of parameters for the kernel

        Returns
        -------
        np.ndarray
            negative ln-concentrated likelihood
        """

        params = np.atleast_2d(params)
        num_params = params.shape[1]
        nll = np.zeros(params.shape[0])

        for i in range(params.shape[0]):

            # for optimization every row is a parameter set
            if self.noise is None:
                param = params[i, 0: num_params - 1]
                noise_sigma = params[i, -1]
            else:
                param = params[i, :]
                noise_sigma = self.noise / self.yh_std

            # calculate the covariance matrix
            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled,
                            param) + noise_sigma**2 * np.eye(self._num_xh)
            L = cholesky(K)
            # Step 1: estimate beta, which is the coefficient of basis function
            # f, basis function
            # f = self.predict_lf(self.sample_xh)
            # alpha = K^(-1) * Y
            alpha = solve(L.T, solve(L, self.sample_yh_scaled))
            # K^(-1)f
            KF = solve(L.T, solve(L, self.f))
            # cholesky decomposition for (F^T *K^(-1)* F)
            ld = cholesky(np.dot(self.f.T, KF))
            # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
            beta = solve(ld.T, solve(ld, np.dot(self.f.T, alpha)))

            # step 2: estimate sigma2
            # gamma = 1/n * (Y - F * beta)^T * K^(-1) * (Y - F * beta)
            gamma = solve(L.T, solve(
                L, (self.sample_yh_scaled - np.dot(self.f, beta))))
            sigma2 = np.dot((self.sample_yh_scaled - np.dot(self.f, beta)).T,
                            gamma) / self._num_xh

            # step 3: calculate the log likelihood
            if self.noise == 0.0:
                logp = -0.5 * self._num_xh * \
                    np.log(sigma2) - np.sum(np.log(np.diag(L)))
            else:
                logp = -0.5 * self._num_xh * \
                    sigma2 - np.sum(np.log(np.diag(L)))

            nll[i] = -logp.ravel()

        return nll

    def _update_parameters(self) -> None:
        """Update parameters of the model"""
        # update parameters with optimized hyper-parameters
        if self.noise is None:
            self.noise = self.opt_param[-1]*self.yh_std
            self.kernel.set_params(self.opt_param[:-1])
        else:
            self.kernel.set_params(self.opt_param)
        # get the kernel matrix
        self.K = self.kernel.get_kernel_matrix(
            self.sample_xh_scaled, self.sample_xh_scaled) + \
            (self.noise/self.yh_std)**2 * np.eye(self._num_xh)

        self.L = cholesky(self.K)

        # step 1: get the optimal beta
        # alpha = K^(-1) * Y
        self.alpha = solve(self.L.T, solve(self.L, self.sample_yh_scaled))
        # K^(-1)f
        self.KF = solve(self.L.T, solve(self.L, self.f))
        self.ld = cholesky(np.dot(self.f.T, self.KF))
        # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
        self.beta = solve(self.ld.T, solve(
            self.ld, np.dot(self.f.T, self.alpha)))

        # step 2: get the optimal sigma2
        self.gamma = solve(self.L.T, solve(
            self.L, (self.sample_yh_scaled - np.dot(self.f, self.beta))))
        self.sigma2 = np.dot((self.sample_yh_scaled -
                             np.dot(self.f, self.beta)).T,
                             self.gamma) / self._num_xh

        # step 3: get the optimal log likelihood
        self.logp = (-0.5 * self._num_xh * self.sigma2 -
                     np.sum(np.log(np.diag(self.L)))).item()

    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict the low-fidelity responses

        Parameters
        ----------
        X : np.ndarray
            test samples

        Returns
        -------
        np.ndarray
            predicted responses of low-fidelity
        """
        return self.lf_model.predict(X)

    def normalize_input(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize the input according to the design space

        Parameters
        ----------
        inputs : np.ndarray
            input samples with shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            normalized samples in to the design space
        """

        return (inputs - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalize_hf_output(self, outputs: np.ndarray) -> np.ndarray:

        self.yh_mean = np.mean(outputs)
        self.yh_std = np.std(outputs)

        return (outputs - self.yh_mean) / self.yh_std

    def _basis_function(self, X: np.ndarray,
                        poly_order: str = "linear") -> np.ndarray:
        """Calculate the basis function

        Parameters
        ----------
        X : np.ndarray
            sample points
        poly_order : str, optional
            order of polynomial, by default "linear"

        Returns
        -------
        np.ndarray
            basis function
        """
        # get the prediction of low-fidelity at high-fidelity locations
        f = self.predict_lf(X)
        # scale the low-fidelity prediction to the same scale as high-fidelity
        f = (f-self.yh_mean)/self.yh_std

        if poly_order == "ordinary":
            # ordinary polynomial (it retrieves back to single fidelity)
            f = np.ones((f.shape[0], 1))
        elif poly_order == "linear_without_const":
            # assemble the basis function without the first column
            f = f
        elif poly_order == "linear":
            # assemble the basis function by having the first column as 1
            f = np.hstack((np.ones((f.shape[0], 1)), f))
        elif poly_order == "quadratic":
            # assemble the basis function with the first column as 1
            f = np.hstack((np.ones((f.shape[0], 1)), f, f**2))
        elif poly_order == "cubic":
            # assemble the basis function with the first column as 1
            f = np.hstack((np.ones((f.shape[0], 1)), f, f**2, f**3))

        else:
            raise ValueError("Invalid polynomial order")

        return f

    @property
    def _get_lf_model(self) -> Any:
        """Get the low-fidelity model

        Returns
        -------
        Any
            low-fidelity model instance
        """

        return self.lf_model

    @property
    def _num_xh(self) -> int:
        """Return the number of high-fidelity samples

        Returns
        -------
        int
            #high-fidelity samples
        """
        return self.sample_xh.shape[0]

    @property
    def _num_xl(self) -> int:
        """Return the number of low-fidelity samples

        Returns
        -------
        int
            #low-fidelity samples
        """
        return self.sample_xl.shape[0]

    @property
    def _get_sample_hf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xh

    @property
    def _get_sample_lf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xl
