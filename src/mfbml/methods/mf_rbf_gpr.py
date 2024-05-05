import time
from typing import Any, Tuple

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# local modules
from .rbf_kernel import RBF
from .rbf_regressor import RBFKernelRegression


class MFRBFGPR:
    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 0,
        kernel: Any = None,
        noise_prior: float = None,
        seed: int = 42,
    ) -> None:

        np.random.seed(seed)
        self.bounds = design_space
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]

        # get the noise level
        self.noise = noise_prior

        # define kernel
        if kernel is None:
            self.kernel = RBF(theta=np.ones(self.num_dim))
        else:
            self.kernel = kernel

        # define the lf model
        self.lf_model = RBFKernelRegression(design_space=self.bounds,
                                            params_optimize=True,
                                            noise_data=True,
                                            optimizer_restart=5,
                                            seed=seed)

    def train(self, samples: dict, responses: dict) -> None:
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
        self.sample_xh = samples["hf"]
        self.sample_xl = samples["lf"]
        self.sample_xh_scaled = self.normalize_input(self.sample_xh)
        self.sample_xl_scaled = self.normalize_input(self.sample_xl)

        # get responses and normalize them
        self.sample_yh = responses["hf"]
        self.sample_yl = responses["lf"]
        self.sample_yh_scaled = self.normalize_hf_output(self.sample_yh)
        self.sample_yl_scaled = (self.sample_yl - self.yh_mean) / self.yh_std
        # rbf surrogate model would normalize the inputs directly
        self.lf_model.train(samples["lf"], responses["lf"])
        # prediction of low-fidelity at high-fidelity locations
        self.f = self._basis_function(self.sample_xh)
        # optimize the hyper parameters of kernel
        self._optimize_parameters()
        # update parameters
        self._update_parameters()

    def predict(self,
                X: np.ndarray,
                return_std: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:
        # normalize the input
        sample_new = self.normalize_input(X)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_xh_scaled, sample_new)
        # calculate the predicted mean
        f = self._basis_function(X)
        # get the mean
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        fmean = (fmean * self.yh_std + self.yh_mean).reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean.reshape(-1, 1)
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

    def _logLikelihood(self, params):

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
            logp = -0.5 * self._num_xh * sigma2 - np.sum(np.log(np.diag(L)))

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
        self.sigma2 = np.dot((self.sample_yh_scaled - np.dot(self.f, self.beta)).T,
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

        return (inputs - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalize_hf_output(self, outputs: np.ndarray) -> np.ndarray:

        self.yh_mean = np.mean(outputs)
        self.yh_std = np.std(outputs)

        return (outputs - self.yh_mean) / self.yh_std

    def _basis_function(self, x: np.ndarray) -> np.ndarray:
        """Calculate the basis function

        Parameters
        ----------
        x : np.ndarray
            sample points

        Returns
        -------
        np.ndarray
            basis function
        """
        # get the prediction of low-fidelity at high-fidelity locations
        f = self.predict_lf(x)
        f = (f-self.yh_mean)/self.yh_std
        # assemble the basis function by having the first column as 1
        f = np.hstack((np.ones((f.shape[0], 1)), f))

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
        return self.lf_model._num_samples

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
