import time
from typing import Any, Tuple

import numpy as np
from mfpml.models.kernels import RBF
from mfpml.models.rbf import RBFSurrogate
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize


class MFRBFKriging:
    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 0,
        kernel: Any = None,
    ) -> None:

        self.bounds = design_space
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]

        # define kernel
        if kernel is None:
            self.kernel = RBF(theta=np.zeros(self.num_dim))
        else:
            self.kernel = kernel

        # define the lf model
        self.lf_model = RBFSurrogate(design_space=self.bounds)

    def train(self, samples: dict, responses: dict) -> None:
        """Train the hierarchical Kriging model

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
        # train the low-fidelity model
        self._train_lf(samples["lf"], responses["lf"])
        # train high-fidelity model, it will be trained at child-class
        self._train_hf(samples["hf"], responses["hf"])

    def _train_hf(self,
                  sample_xh: np.ndarray,
                  sample_yh: np.ndarray) -> None:

        self.sample_xh = sample_xh
        self.sample_xh_scaled = self.normalize_input(self.sample_xh)
        self.sample_yh = sample_yh.reshape(-1, 1)
        # normalize the output
        self.sample_yh_scaled = self.normalize_hf_output(self.sample_yh)
        self.sample_yl_scaled = (self.sample_yl - self.yh_mean) / self.yh_std
        # prediction of low-fidelity at high-fidelity locations
        f = self.predict_lf(self.sample_xh)
        self.f = (f-self.yh_mean)/self.yh_std
        # optimize the hyper parameters
        self._optHyp()
        # update kernel parameters
        self.kernel.set_params(self.opt_param)
        # update kriging parameters
        self._update_parameters()

    def predict(self,
                x_predict: np.ndarray,
                return_std: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:
        # normalize the input
        sample_new = self.normalize_input(x_predict)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_xh_scaled, sample_new)
        # calculate the normalized predicted mean
        f = (self.predict_lf(x_predict) - self.yh_mean) / self.yh_std
        # get the mean
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        # scaled the mean back to original scale
        fmean = fmean * self.yh_std + self.yh_mean
        # calculate the standard deviation
        if not return_std:
            return fmean.reshape(-1, 1)
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            mse = self.sigma2 * (
                1
                - np.diag(knew.T.dot(delta))
                + np.diag(
                    np.dot(
                        (knew.T.dot(self.KF) - f),
                        (knew.T.dot(self.KF) - f).T,
                    )
                )
                / self.f.T.dot(self.KF)
            )
            std = np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)
            std = std * self.yh_std

            return fmean, std

    def _optHyp(self) -> None:

        if self.optimizer is None:
            n_trials = self.optimizer_restart + 1
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    self.kernel._get_low_bound,
                    self.kernel._get_high_bound,
                    self.kernel._get_num_para,
                )
                optRes = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=self.kernel._get_bounds_list,
                )
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=self.kernel._get_num_para,
                design_space=self.kernel._get_bounds,
            )
            opt_param = optRes["best_x"]
        self.opt_param = opt_param

    def _logLikelihood(self, params):

        params = np.atleast_2d(params)
        nll = np.zeros(params.shape[0])

        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            param = params[i, :]
            # calculate the covariance matrix
            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled,
                            param)
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
            # beta = (F^T *K^(-1)* F)^(-1) * F^T *K^(-1) * Y
            beta = solve(ld.T, solve(ld, np.dot(self.f.T, alpha)))

            # step 2: estimate sigma2
            # gamma = 1/n * (Y - F * beta)^T * K^(-1) * (Y - F * beta)
            gamma = solve(L.T, solve(
                L, (self.sample_yh_scaled - np.dot(self.f, beta))))
            sigma2 = np.dot((self.sample_yh_scaled - np.dot(self.f, beta)).T,
                            gamma) / self._num_xh

            # step 3: calculate the log likelihood
            logp = -0.5 * self._num_xh * \
                np.log(sigma2) - np.sum(np.log(np.diag(L)))
            nll[i] = -logp.ravel()

        return nll

    def _update_parameters(self) -> None:
        """Update parameters of the model"""
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(
            self.sample_xh_scaled, self.sample_xh_scaled)
        self.L = cholesky(self.K)

        # step 1: get the optimal beta
        # f, basis function
        # self.f = self.predict_lf(self.sample_xh)
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
        self.logp = (-0.5 * self._num_xh * np.log(self.sigma2) -
                     np.sum(np.log(np.diag(self.L)))).item()

    def _update_optimizer_hf(self, optimizer: Any) -> None:
        """Change the optimizer for high-fidelity hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def _train_lf(self,
                  sample_xl: np.ndarray,
                  sample_yl: np.ndarray) -> None:

        # rbf surrogate model would normalize the inputs directly
        self.lf_model.train(sample_xl, sample_yl)
        # normalize the input
        self.sample_xl = sample_xl
        self.sample_xl_scaled = self.normalize_input(sample_xl)
        self.sample_yl = sample_yl

    def predict_lf(self, test_xl: np.ndarray) -> np.ndarray:

        return self.lf_model.predict(test_xl)

    def normalize_input(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize the input into [0, 1] according to the bounds

        Parameters
        ----------
        inputs : np.ndarray
            sample points

        Returns
        -------
        np.ndarray
            normalized sample points
        """

        return (inputs - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalize_hf_output(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize the output into normal distribution

        Parameters
        ----------
        outputs : np.ndarray
            responses of high-fidelity

        Returns
        -------
        np.ndarray
            normalized responses
        """
        # calculate the mean and std of hf responses
        self.yh_mean = np.mean(outputs)
        self.yh_std = np.std(outputs)

        return (outputs - self.yh_mean) / self.yh_std

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
