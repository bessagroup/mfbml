# ------------------ Beginning of Reference Python Module ---------------------
"""The proposed DNN-LR-BNN framework is implemented in this module.


Classes
-------
ClassOne
    A one-line summary of ClassOne contained in this module.
ClassTwo
    A one-line summary of ClassTwo contained in this module.

Functions
---------
function_one
    A one-line summary of function_one contained in this module.
function_two
    A one-line summary of function_two contained in this module.
"""

#
#                                                                       Modules
# =============================================================================
# standard library
from typing import Any, Dict, List, Tuple

# third party modules
import numpy as np
import torch
from torch import Tensor
from scipy.optimize import minimize
# local modules
from mfbml.methods.bayes_neural_nets import BNNWrapper
from mfbml.methods.deep_neural_nets import LFDNN

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class DNNLinearRegressionBNN:
    """A class for the multi-fidelity DNN-BNN framework, the first step of the
    multi-fidelity framework is to create a low-fidelity model the low-fidelity
    model is a DNN the DNN is created using the LFDNN; the second step of the
    multi-fidelity framework is to create a high-fidelity model using Bayesian
    neural network (BNN), the BNN is created using the BNNWrapper.
    """

    def __init__(self,
                 design_space: torch.Tensor,
                 lf_configure: Dict,
                 hf_configure: Dict,
                 beta_optimize: bool = True,
                 lf_order: int = 1,
                 beta_bounds: List = [1e-2, 1e1],
                 optimizer_restart: int = 20,
                 discrepancy_normalization: str = "hf") -> None:
        """initialize the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        design_space: torch.Tensor
            the design space of the problem(for scaling the input data)
        lf_configure : dict
            a dictionary containing the configuration of the low-fidelity
        hf_configure : dict
            a dictionary containing the configuration of the high-fidelity
        beta_optimize : bool, optional
            whether to optimize the beta or not, by default True
        beta_bounds : list, optional
            the bounds of the beta, by default [1e-2, 1e1]
        optimizer_restart : int, optional
            the number of restarts for the optimizer, by default 20
        discrepancy_normalization : str, optional
            the normalization method for the discrepancy, by default "hf"
        """
        # get the design space of this problem
        self.design_space = design_space
        # record the configuration of the low-fidelity model
        self.lf_configure = lf_configure
        self.hf_configure = hf_configure

        # record beta optimize or not
        self.beta_optimize = beta_optimize
        self.lf_order = lf_order
        self.optimizer_restart = optimizer_restart
        self.beta = np.ones(lf_order+1)
        # create the beta bounds
        self.beta_low_bounds = [beta_bounds[0] for i in range(lf_order+1)]
        self.beta_high_bounds = [beta_bounds[1] for i in range(lf_order+1)]
        self.beta_bounds = (self.beta_low_bounds, self.beta_high_bounds)

        # record the discrepancy normalization method
        self.discrepancy_normalization = discrepancy_normalization

        # create the low-fidelity model
        self.define_lf_model(lf_model=None)
        # create the high-fidelity model
        self.define_hf_model(hf_model=None)

    def define_lf_model(self, lf_model: Any = None) -> None:
        """create the low-fidelity model, it can be defined by using passing a
        dictionary containing the configuration of the low-fidelity model or
        passing a LFDNN object directly from this function.
        """
        if lf_model is None:
            # create the low-fidelity model
            self.lf_model = LFDNN(
                in_features=self.lf_configure["in_features"],
                hidden_features=self.lf_configure["hidden_features"],
                out_features=self.lf_configure["out_features"],
                activation=self.lf_configure["activation"],
                optimizer=self.lf_configure["optimizer"],
                lr=self.lf_configure["lr"],
                weight_decay=self.lf_configure["weight_decay"],
                loss=self.lf_configure["loss"])
        else:
            self.lf_model = lf_model

    def define_hf_model(self, hf_model: Any = None) -> None:
        """create the high-fidelity model, it can be defined by using passing a
        dictionary containing the configuration of the high-fidelity model or
        passing a BNNWrapper object directly from this function.

        """
        if hf_model is None:
            # create the high-fidelity model
            self.hf_model = BNNWrapper(
                in_features=self.hf_configure["in_features"],
                hidden_features=self.hf_configure["hidden_features"],
                out_features=self.hf_configure["out_features"],
                activation=self.hf_configure["activation"],
                lr=self.hf_configure["lr"],
                sigma=self.hf_configure["sigma"],
            )
        else:
            self.hf_model = hf_model

    def train(self,
              X: List,
              Y: List,
              lf_train_config: Dict = {"batch_size": None,
                                       "num_epochs": 1000,
                                       "print_iter": 100,
                                       "data_split": True,
                                       },
              hf_train_config: Dict = {"num_epochs": 10000,
                                       "sample_freq": 100,
                                       "print_info": True,
                                       "burn_in_epochs": 1000}) -> None:
        """train the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        X : List
            a dictionary containing the low-fidelity and high-fidelity samples,
            original scale is expected, with the key "lf" and "hf"
        Y : List
            a dictionary containing the low-fidelity and high-fidelity
            responses, original scale is expected, with the key "lf" and "hf"
        lf_train_config : dict, optional
            low fidelity configuration, by default {"batch_size": None,
            "num_epochs": 1000, "print_iter": 100}
        hf_train_config : dict, optional
            high fidelity configuration, by default {"num_epochs": 10000,
            "sample_freq": 100, "print_info": True, "burn_in_epochs": 1000}
        """
        # get the low-fidelity samples
        self.lf_samples = X[1]
        self.lf_samples_scaled = self.normalize_inputs(self.lf_samples)
        # get the high-fidelity samples
        self.hf_samples = X[0]
        self.hf_samples_scaled = self.normalize_inputs(self.hf_samples)
        # get the low-fidelity responses
        self.lf_responses = Y[1]
        self.lf_responses_scaled = self.normalize_lf_output(self.lf_responses)
        # get the high-fidelity responses
        self.hf_responses = Y[0]
        # train the low-fidelity model
        self.train_lf_model(X=self.lf_samples_scaled,
                            Y=self.lf_responses_scaled,
                            batch_size=lf_train_config["batch_size"],
                            num_epochs=lf_train_config["num_epochs"],
                            print_iter=lf_train_config["print_iter"],
                            data_split=lf_train_config["data_split"])
        # get the prediction of the low-fidelity model at original scale
        lf_hf_samples = self.predict_lf(
            self.hf_samples, output_format="torch")
        # optimize the beta (in the original scale)
        if self.beta_optimize:
            self.beta = self._beta_optimize()

        if self.discrepancy_normalization == "hf":
            # normalize the hf responses (not used yet)
            self.hf_responses_scaled = self.normalize_hf_output(
                self.hf_responses)
            # scale the noise for HF model
            self.hf_model.sigma = self.hf_model.sigma / self.yh_std.numpy()
            # scale it to the hf scale
            # get the difference between the hf and lf samples
            if self.lf_order == 1:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[1] * lf_hf_samples - self.beta[0]
            elif self.lf_order == 2:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[2] * lf_hf_samples**2 - \
                    self.beta[1] * lf_hf_samples - self.beta[0]
            elif self.lf_order == 3:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[3] * lf_hf_samples**3 - \
                    self.beta[2] * lf_hf_samples**2 - \
                    self.beta[1] * lf_hf_samples - self.beta[0]
            else:
                raise ValueError(
                    "The order of the low-fidelity model is not supported")
            # transfer to torch tensor
            dis_hf_lf_samples = torch.Tensor(dis_hf_lf_samples)
            # scale the discrepancy according to hf
            dis_hf_lf_samples = (dis_hf_lf_samples-self.yh_mean)/self.yh_std

        elif self.discrepancy_normalization == "diff":
            # get discrepancy between HF and LF samples in the original scale
            if self.lf_order == 1:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[1]*lf_hf_samples - self.beta[0]
            elif self.lf_order == 2:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[2]*lf_hf_samples**2 - \
                    self.beta[1]*lf_hf_samples - self.beta[0]
            elif self.lf_order == 3:
                dis_hf_lf_samples = self.hf_responses - \
                    self.beta[3]*lf_hf_samples**3 - \
                    self.beta[2]*lf_hf_samples**2 - \
                    self.beta[1]*lf_hf_samples - self.beta[0]
            else:
                raise ValueError(
                    "The order of the low-fidelity model is not supported")

            dis_hf_lf_samples = self.normalize_diff_output(
                dis_hf_lf_samples)
            dis_hf_lf_samples = torch.Tensor(dis_hf_lf_samples)
            self.hf_model.sigma = self.hf_model.sigma / self.diff_std.numpy()

        # train the high-fidelity model (normalized discrepancy)
        self.train_hf_model(X=self.hf_samples_scaled,
                            Y=dis_hf_lf_samples,
                            num_epochs=hf_train_config["num_epochs"],
                            sample_freq=hf_train_config["sample_freq"],
                            verbose=hf_train_config["print_info"],
                            burn_in_epochs=hf_train_config["burn_in_epochs"])

    def predict(self,
                X: Tensor) -> Tuple[np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
        """predict the high fidelity output of the MF-DNN-BNN framework

        Parameters
        ----------
        X : torch.Tensor
            test input data

        Returns
        -------
        Tuple
            predictive mean, epistemic uncertainty, total uncertainty,
            aleatoric uncertainty
        """

        if self.discrepancy_normalization == "diff":
            # get the low-fidelity model prediction
            lf_y = self.predict_lf(X, output_format="numpy")
            # scale the input data
            x_scale = self.normalize_inputs(X)
            # get the high-fidelity model prediction
            hy_pred, epistemic, total_unc, aleatoric = self.hf_model.predict(
                x_scale)

            # scale the discrepancy to the original scale
            hy_pred = hy_pred * self.diff_std.numpy() + self.diff_mean.numpy()

            # get the final prediction y = beta1*lf_y + hy_pred + beta0
            if self.lf_order == 1:
                y = self.beta[1]*lf_y + hy_pred + self.beta[0]
            elif self.lf_order == 2:
                y = self.beta[2]*lf_y**2 + self.beta[1] * \
                    lf_y + hy_pred + self.beta[0]
            elif self.lf_order == 3:
                y = self.beta[3]*lf_y**3 + self.beta[2]*lf_y**2 + \
                    self.beta[1]*lf_y + hy_pred + self.beta[0]
            else:
                raise ValueError(
                    "The order of the low-fidelity model is not supported")

            # uncertainties
            epistemic = epistemic * self.diff_std.numpy()
            total_unc = total_unc * self.diff_std.numpy()
            aleatoric = aleatoric * self.diff_std.numpy()

        elif self.discrepancy_normalization == "hf":

            # get the low-fidelity model prediction
            lf_y = self.predict_lf(X, output_format="numpy")
            # scale the input data
            x_scale = self.normalize_inputs(X)
            # get the high-fidelity model prediction
            hy_pred, epistemic, total_unc, aleatoric = self.hf_model.predict(
                x_scale)
            # scale the discrepancy to the original scale
            hy_pred = hy_pred * self.yh_std.numpy() + self.yh_mean.numpy()

            # get the final prediction at the scaled scale
            if self.lf_order == 1:
                y = self.beta[1]*lf_y + hy_pred + self.beta[0]
            elif self.lf_order == 2:
                y = self.beta[2]*lf_y**2 + \
                    self.beta[1]*lf_y + hy_pred + self.beta[0]
            elif self.lf_order == 3:
                y = self.beta[3]*lf_y**3 + \
                    self.beta[2]*lf_y**2 + \
                    self.beta[1]*lf_y + hy_pred + self.beta[0]
            else:
                raise ValueError(
                    "The order of the low-fidelity model is not supported")

            epistemic = epistemic * self.yh_std.numpy()
            total_unc = total_unc * self.yh_std.numpy()
            aleatoric = aleatoric * self.yh_std.numpy()

        return y, epistemic, total_unc, aleatoric

    def predict_lf(self, X: torch.Tensor,
                   output_format: str = "torch") -> torch.Tensor | np.ndarray:
        """predict the low fidelity output of the MF-DNN-BNN framework

        Parameters
        ----------
        X : torch.Tensor
            test input data

        Returns
        -------
        torch.Tensor
            test output data

        """
        x_scaled = self.normalize_inputs(X)
        # get the low-fidelity model prediction
        lf_y_scaled = self.lf_model.forward(x_scaled)
        # scale back to the original scale
        lf_y = lf_y_scaled * self.yl_std + self.yl_mean

        if output_format == "torch":
            return lf_y
        elif output_format == "numpy":
            return lf_y.detach().numpy()

        return lf_y

    def train_lf_model(self,
                       X: torch.Tensor,
                       Y: torch.Tensor,
                       batch_size: int = None,  # type: ignore
                       num_epochs: int = 10000,
                       print_iter: int = 100,
                       data_split: bool = False
                       ) -> None:
        """train the low-fidelity model

        Parameters
        ----------
        X : torch.Tensor
            input data of the low-fidelity model
        Y : torch.Tensor
            output data of the low-fidelity model
        batch_size : int, optional
            batch size, by default None
        num_epochs : int, optional
            number epochs, by default 1000
        print_iter : int, optional
            print iteration, by default 100
        """
        self.lf_model.train(X=X,
                            Y=Y,
                            batch_size=batch_size,
                            num_epoch=num_epochs,
                            print_iter=print_iter,
                            data_split=data_split)

    def train_hf_model(self,
                       X: torch.Tensor,
                       Y: torch.Tensor,
                       num_epochs: int = None,  # type: ignore
                       sample_freq: int = 10000,
                       verbose: bool = True,
                       burn_in_epochs: int = 1000
                       ) -> None:
        """train the high-fidelity model

        Parameters
        ----------
        X : torch.Tensor
            input data of the high-fidelity model
        Y : torch.Tensor
            output data of the high-fidelity model
        num_epochs : int, optional
            number epochs, by default None
        sample_freq : int, optional
            sample frequency, by default 10000
        verbose : bool, optional
            print information of not, by default True
        burn_in_epochs : int, optional
            burn in epochs, by default 1000
        """
        self.hf_model.train(X=X,
                            Y=Y,
                            num_epochs=num_epochs,
                            sample_freq=sample_freq,
                            verbose=verbose,
                            burn_in_epochs=burn_in_epochs)

    def _beta_optimize(self) -> np.ndarray:
        """optimize the beta, the beta is the parameter determining how much
        low-fidelity model is trusted in the multi-fidelity DNN-BNN framework.
        The beta is optimized by minimizing the error between the hf model
        and lf model.

        Returns
        -------
        np.ndarray
            the optimized beta
        """

        # optimize the beta
        n_trials = self.optimizer_restart + 1
        optimum_value = float("inf")
        for _ in range(n_trials):
            x0 = np.random.uniform(
                self.beta_low_bounds,
                self.beta_high_bounds,
                size=(self.lf_order+1),
            )
            optRes = minimize(
                self._eval_error,
                x0=x0,
                method="L-BFGS-B",
                bounds=np.array([self.beta_bounds]).T,
            )
            if optRes.fun < optimum_value:
                optimum_value = optRes.fun
                beta = optRes.x

        return beta

    def _eval_error(self, beta: np.ndarray) -> np.ndarray:
        """calculate error between the high-fidelity and low-fidelity model

        Parameters
        ----------
        beta : np.ndarray
            parameter determining how much the low-fidelity model is trusted

        Returns
        -------
        np.ndarray
            the error between the high-fidelity and low-fidelity model
        """
        # get the low-fidelity model prediction of the high-fidelity samples
        hf_responses = self.hf_responses.clone().detach().numpy()
        lf_responses = self.predict_lf(self.hf_samples, output_format="numpy")
        beta = np.tile(beta, (hf_responses.shape[0], 1))

        # calculate the error between the high-fidelity and low-fidelity model
        if self.lf_order == 1:
            error = (beta[:, 1] * lf_responses.ravel() -
                     hf_responses.ravel() + beta[:, 0].ravel())
        elif self.lf_order == 2:
            error = (beta[:, 2] * lf_responses.ravel()**2 +
                     beta[:, 1] * lf_responses.ravel() -
                     hf_responses.ravel() + beta[:, 0].ravel())
        elif self.lf_order == 3:
            error = (beta[:, 3] * lf_responses.ravel()**3 +
                     beta[:, 2] * lf_responses.ravel()**2 +
                     beta[:, 1] * lf_responses.ravel() -
                     hf_responses.ravel() + beta[:, 0].ravel())
        else:
            raise ValueError(
                "The order of the low-fidelity model is not supported")

        # calculate the summation of the error
        sum_error = np.sum(error**2)

        return sum_error

    def normalize_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """normalize the input data

        Parameters
        ----------
        X : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            normalized input data
        """
        X = (X - self.design_space[:, 0]) / \
            (self.design_space[:, 1] - self.design_space[:, 0])

        return X

    def normalize_lf_output(self, Y: torch.Tensor) -> torch.Tensor:
        """normalize the output data of the low-fidelity model

        Parameters
        ----------
        Y : torch.Tensor
            output data of the low-fidelity model

        Returns
        -------
        torch.Tensor
            normalized output data of the low-fidelity model
        """
        self.yl_mean = torch.mean(Y, axis=0)
        self.yl_std = torch.std(Y, axis=0)
        Y = (Y - self.yl_mean) / self.yl_std

        return Y

    def normalize_diff_output(self, Y: torch.Tensor) -> torch.Tensor:
        """normalize the output data of the high-fidelity model

        Parameters
        ----------
        Y : torch.Tensor
            output data of the high-fidelity model

        Returns
        -------
        torch.Tensor
            normalized output data of the high-fidelity model
        """
        self.diff_mean = torch.mean(Y, axis=0).detach()
        self.diff_std = torch.std(Y, axis=0).detach()
        Y = (Y - self.diff_mean) / self.diff_std

        return Y

    def normalize_hf_output(self, Y: torch.Tensor) -> torch.Tensor:
        """normalize the output data of the high-fidelity model

        Parameters
        ----------
        Y : torch.Tensor
            output data of the high-fidelity model

        Returns
        -------
        torch.Tensor
            normalized output data of the high-fidelity model
        """
        self.yh_mean = torch.mean(Y, axis=0)
        self.yh_std = torch.std(Y, axis=0)
        Y = (Y - self.yh_mean) / self.yh_std

        return Y
