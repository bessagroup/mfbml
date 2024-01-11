# this script is used for the mf_dnn_bnn framework
# standard library
from typing import Any, Tuple

# third party modules
import numpy as np
import torch
from torch import nn as nn

# local modules
from mfbml.methods.bnn import BNNWrapper
# import local modules
from mfbml.methods.dnn import LFDNN


class SequentialMFBNN:
    """
    A class for multi-fidelity Bayesian neural network (BNN) framework, the
    multi-fidelity framework is to create a low-fidelity model the low-fidelity
    model is a DNN the DNN is created using the LFDNN; the second step of the
    multi-fidelity framework is to create a high-fidelity model using Baysian 
    neural network (BNN), the BNN is created using the BNNWrapper. In this class
    the two models are trained sequentially.
    """

    def __init__(self,
                 lf_configure: dict,
                 hf_configure: dict,) -> None:
        """initialize the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        lf_configure : dict
            a dictionary containing the configuration of low-fidelity model
        hf_configure : dict
            a dictionary containing the configuration of high-fidelity model
        """
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
              samples: dict,
              responses: dict,
              lf_train_config: dict = {"batch_size": None,
                                       "num_epochs": 50000,
                                       "print_iter": 1000},
              hf_train_config: dict = {"num_epochs": 10000,
                                       "sample_freq": 100,
                                       "print_info": True,
                                       "burn_in_epochs": 1000}):
        """train the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        samples : dict
            a dictionary containing the low-fidelity and high-fidelity samples
        responses : dict
            a dictionary containing the low-fidelity and high-fidelity responses
        lf_train_config : dict, optional
            low fidelity training configuration, by default {"batch_size": None,
            "num_epochs": 1000, "print_iter": 100}
        hf_train_config : dict, optional
            high fidelity training configuration, by default {"num_epochs": 10000,
            "sample_freq": 100, "print_info": True, "burn_in_epochs": 1000}
        """
        # get the low-fidelity samples
        self.lf_samples = samples["lf"]
        # get the high-fidelity samples
        self.hf_samples = samples["hf"]
        # get the low-fidelity responses
        self.lf_responses = responses["lf"]
        # get the high-fidelity responses
        self.hf_responses = responses["hf"]
        # receive the low-fidelity training configuration
        self.lf_training_configure = lf_train_config
        # receive the high-fidelity training configuration
        self.hf_training_configure = hf_train_config

        # train the low-fidelity model
        self.train_lf_model(x=samples["lf"],
                            y=responses["lf"],
                            batch_size=lf_train_config["batch_size"],
                            num_epochs=lf_train_config["num_epochs"],
                            print_iter=lf_train_config["print_iter"])

        # get the low-fidelity model prediction of the high-fidelity samples
        lf_hf_samples = self.lf_model.forward(samples["hf"])

        # concatenate xh and yh
        xh_ylf = torch.concatenate((samples["hf"], lf_hf_samples), dim=1)

        # check the input dimension of the high-fidelity model
        self._check_hf_input_dimension()
        # train the high-fidelity model
        self.train_hf_model(x=xh_ylf,
                            y=responses["hf"],
                            num_epochs=hf_train_config["num_epochs"],
                            sample_freq=hf_train_config["sample_freq"],
                            print_info=hf_train_config["print_info"],
                            burn_in_epochs=hf_train_config["burn_in_epochs"])

    def predict(self, x: torch.Tensor):
        """predict the high fidelity output of the multi-fidelity DNN-BNN framework

        Parameters
        ----------
        x : torch.Tensor
            test input data

        Returns
        -------
        _type_
            _description_
        """
        # get the low-fidelity model prediction
        lf_y = self.lf_model.forward(x)
        # having the xh_ylf
        xh_ylf = torch.concatenate((x, lf_y), dim=1)
        # get the high-fidelity model prediction
        hf_mean, epistemic, total_unc, aleatoric = self.hf_model.predict(
            xh_ylf)

        return hf_mean, epistemic, total_unc, aleatoric

    def train_lf_model(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       batch_size: int = None,  # type: ignore
                       num_epochs: int = 10000,
                       print_iter: int = 100
                       ) -> None:
        """train the low-fidelity model

        Parameters
        ----------
        x : torch.Tensor
            input data of the low-fidelity model
        y : torch.Tensor
            output data of the low-fidelity model
        batch_size : int, optional
            batch size, by default None
        num_epochs : int, optional
            number epochs, by default 1000
        print_iter : int, optional
            print iteration, by default 100
        """
        self.lf_model.train(x=x,
                            y=y,
                            batch_size=batch_size,
                            num_epoch=num_epochs,
                            print_iter=print_iter)

    def train_hf_model(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       num_epochs: int = None,  # type: ignore
                       sample_freq: int = 10000,
                       print_info: bool = True,
                       burn_in_epochs: int = 1000
                       ) -> None:
        """train the high-fidelity model

        Parameters
        ----------
        x : torch.Tensor
            input data of the high-fidelity model
        y : torch.Tensor
            output data of the high-fidelity model
        num_epochs : int, optional
            number epochs, by default None
        sample_freq : int, optional
            sample frequency, by default 10000
        print_info : bool, optional
            print information of not, by default True
        burn_in_epochs : int, optional
            burn in epochs, by default 1000
        """
        self.hf_model.train(x=x,
                            y=y,
                            num_epochs=num_epochs,
                            sample_freq=sample_freq,
                            print_info=print_info,
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
