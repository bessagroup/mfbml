# ------------------ Beginning of Reference Python Module ---------------------
""" Module for Single-fidelity Bayesian Neural Networks

This module contains the classes and functions for training single-fidelity
Bayesian Neural Networks (BNNs) using PyTorch.

Classes
-------
LinearNet
    A class for defining a linear neural network architecture.
BNNWrapper
    A wrapper class for training a Bayesian Neural Network (BNN) using PyTorch.

"""

#
#                                                                       Modules
# =============================================================================
# standard library modules
from typing import Any, Dict, List, Tuple

# third party modules
import numpy as np
import torch

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


class DNNBNN:
    """
    A class for multi-fidelity Bayesian neural network (BNN) framework, the
    multi-fidelity framework is to create a low-fidelity model the low-fidelity
    model is a DNN the DNN is created using the LFDNN; the second step of the
    multi-fidelity framework is to create a high-fidelity model using Bayesian
    neural network (BNN), the BNN is created using the BNNWrapper. In this
    class the two models are trained sequentially.
    """

    def __init__(self,
                 design_space: torch.Tensor,
                 lf_configure: Dict,
                 hf_configure: Dict,) -> None:
        """initialize the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        lf_configure : Dict
            a dictionary containing the configuration of low-fidelity model
        hf_configure : Dict
            a dictionary containing the configuration of high-fidelity model
        """
        # get the design space of this problem
        self.design_space = design_space
        # record the configuration of the low-fidelity model
        self.lf_configure = lf_configure
        self.hf_configure = hf_configure

        # create the low-fidelity model
        self.define_lf_model()
        # create the high-fidelity model
        self.define_hf_model()

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
                                       "num_epochs": 50000,
                                       "print_iter": 1000,
                                       "data_split": False},
              hf_train_config: Dict = {"num_epochs": 10000,
                                       "sample_freq": 100,
                                       "print_info": True,
                                       "burn_in_epochs": 1000}) -> None:
        """train the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        X : Dict
            a dictionary containing the lf and hf samples
        Y : Dict
            a dictionary containing the lf and hf responses
        lf_train_config : Dict, optional
            low fidelity configuration, by default {"batch_size": None,
            "num_epochs": 1000, "print_iter": 100}
        hf_train_config : Dict, optional
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
        self.hf_responses_scaled = self.normalize_hf_output(self.hf_responses)
        # scale the noise for HF model
        self.hf_model.sigma = self.hf_model.sigma / self.yh_std.numpy()
        # receive the low-fidelity training configuration
        self.lf_training_configure = lf_train_config
        # receive the high-fidelity training configuration
        self.hf_training_configure = hf_train_config

        # train the low-fidelity model
        self.train_lf_model(X=self.lf_samples_scaled,
                            Y=self.lf_responses_scaled,
                            batch_size=lf_train_config["batch_size"],
                            num_epochs=lf_train_config["num_epochs"],
                            print_iter=lf_train_config["print_iter"],
                            data_split=lf_train_config["data_split"])

        # get the low-fidelity model prediction of the high-fidelity samples
        lf_hf_samples = self.predict_lf(
            X=self.hf_samples, output_format="torch")
        # scale the low-fidelity model prediction
        lf_hf_samples = (lf_hf_samples - self.yh_mean) / self.yh_std

        # concatenate xh and yh
        xh_ylf = torch.concatenate(
            (self.hf_samples_scaled, lf_hf_samples), dim=1)

        # check the input dimension of the high-fidelity model
        self._check_hf_input_dimension()
        # train the high-fidelity model
        self.train_hf_model(X=xh_ylf,
                            Y=self.hf_responses_scaled,
                            num_epochs=hf_train_config["num_epochs"],
                            sample_freq=hf_train_config["sample_freq"],
                            verbose=hf_train_config["print_info"],
                            burn_in_epochs=hf_train_config["burn_in_epochs"])

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """predict the high fidelity output of the  Meng's MF-BNN framework

        Parameters
        ----------
        X : torch.Tensor
            test input data

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            high fidelity mean, epistemic uncertainty, total uncertainty,
            aleatoric uncertainty
        """
        # get the low-fidelity model prediction
        lf_y = self.predict_lf(X, output_format="torch")
        # scale the low-fidelity model prediction
        lf_y = (lf_y - self.yh_mean) / self.yh_std
        # scale the test input data
        x_scale = self.normalize_inputs(X)
        # having the xh_ylf
        xh_ylf = torch.concatenate((x_scale, lf_y), dim=1)
        # get the high-fidelity model prediction
        hf_mean, epistemic, total_unc, aleatoric = self.hf_model.predict(
            xh_ylf)

        # scale back to the original scale
        hf_mean = hf_mean * self.yh_std.numpy() + self.yh_mean.numpy()
        epistemic = epistemic * self.yh_std.numpy()
        total_unc = total_unc * self.yh_std.numpy()
        aleatoric = aleatoric * self.yh_std.numpy()

        return hf_mean, epistemic, total_unc, aleatoric

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
                       batch_size: int = None,
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
                            verbose_interval=100,
                            burn_in_epochs=burn_in_epochs)

    def _check_hf_input_dimension(self) -> None:
        """check the input dimension of the high-fidelity model
        """
        # check the input dimension of the high-fidelity model
        if self.hf_configure["in_features"] != \
                self.lf_model.out_features + self.lf_model.in_features:
            raise ValueError(
                "the input dimension of the high-fidelity model should equal"
                "to the sum of the input dimension of the high-fidelity model"
                "and the output dimension of the low-fidelity model")

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

    def normalize_hf_output(self, Y: torch.Tensor) -> torch.Tensor:
        """normalize the output data of the high-fidelity model

        Parameters
        ----------
        y : torch.Tensor
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
