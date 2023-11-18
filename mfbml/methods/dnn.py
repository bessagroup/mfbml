
# this script is used for the mf_dnn_bnn framework
from typing import Any

import torch
from torch import nn as nn


# class for low-fidelity DNN
class MLP(nn.Module):
    def __init__(
        self,
        in_features=1,
        hidden_features=[20, 20, 20],
        out_features=1,
        activation: str = "ReLU",
    ) -> None:
        """
        A simple multi-layer perceptron (MLP) class.

        Parameters
        ----------
        in_features : int, optional
            number of design variables, by default 1
        hidden_features : list, optional
            hidden layer info, by default [20, 20, 20]
        out_features : int, optional
            number of out features, by default 1
        activation : str, optional
            activation, by default "ReLU"
        """
        super().__init__()

        # arguments
        self.in_features = in_features
        self.num_hidden = len(hidden_features)
        self.out_features = out_features
        self.hidden_layers = hidden_features
        self.activation = activation
        # mlp
        self.net = self._create_nn_architecture()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            prediction of the model
        """

        return self.net(x)

    def _create_nn_architecture(self) -> Any:
        """create the nn architecture

        Returns
        -------
        Any
            nn architecture
        """

        # create the first layer
        layers = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layers[0])
        )
        layers.append(self._get_activation())
        for ii in range(1, self.num_hidden):
            layers.append(
                nn.Linear(self.hidden_layers[ii - 1], self.hidden_layers[ii])
            )
            layers.append(self._get_activation())
        # add the last layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.out_features))

        return layers

    def _get_activation(self) -> Any:
        """get activation function according names

        Returns
        -------
        Any
            activation function

        Raises
        ------
        ValueError
            the activation is not implemented in this framework!
        """

        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "Tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                "the activation is not implemented in this framework!"
            )


class LFDNN(MLP):
    """class for training low fidelity deep neural network

    Parameters
    ----------
    MLP : class
        a simple multi-layer perceptron (MLP) class
    """

    def __init__(self,
                 in_features=1,
                 hidden_features=[20, 20, 20],
                 out_features=1,
                 activation: str = "ReLU",
                 optimizer: str = "Adam",
                 lr: float = 0.001,
                 weight_decay: float = 0.01,
                 loss: str = "mse") -> None:
        """initialization of low fidelity deep neural network

        Parameters
        ----------
        in_features : int, optional
            number of in features, by default 1
        hidden_features : list, optional
            hidden features, by default [20, 20, 20]
        out_features : int, optional
            out features, by default 1
        activation : str, optional
            name of activation, by default "ReLU"
        optimizer : str, optional
            name of optimization, by default "Adam"
        lr : float, optional
            learning rate, by default 0.001
        weight_decay : float, optional
            weight decay, by default 0.01
        loss : str, optional
            loss, by default "mse"
        """

        # call the super class
        MLP.__init__(self, in_features, hidden_features,
                     out_features, activation)

        # arguments for optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        # optimizer
        self.optimizer = self._get_optimizer()

        # get the loss function
        self.loss = self._get_loss(loss_name=loss)

    def train(self, x: torch.Tensor,
              y: torch.Tensor,
              batch_size: int | bool = None,  # type: ignore
              num_epoch: int = 1000,
              print_iter: int = 100,) -> None:
        """train the model

        Parameters
        ----------
        x : torch.Tensor
            input data
        y : torch.Tensor
            output data
        batch_size : int, optional
            batch size, by default None
        num_epoch : int, optional
            number of epochs, by default 1000
        """
        # give me the training process code here

        self.num_epoch = num_epoch
        self.batch_size = batch_size

        for epoch in range(self.num_epoch):
            # train the model
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()

            # print the loss to the screen
            if (epoch+1) % print_iter == 0:
                print("epoch: ", epoch, "loss: ", loss.item())

    def _get_optimizer(self) -> Any:
        """get optimizer according names

        Returns
        -------
        Any
            optimizer

        Raises
        ------
        ValueError
            the optimizer is not implemented in this framework!
        """

        if self.optimizer_name == "Adam":

            return torch.optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == "SGD":
            return torch.optim.SGD(self.net.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        else:
            raise ValueError(
                "the optimizer is not implemented in this framework!"
            )

    def _get_loss(self, loss_name) -> nn.MSELoss | nn.L1Loss:
        """get loss function according names

        Returns
        -------
        Any
            loss function

        Raises
        ------
        ValueError
            the loss is not implemented in this framework!
        """

        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(
                "the loss is not implemented in this framework!"
            )

    # write a function to change the optimizer
    def change_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """change the optimizer

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            optimizer
        """
        self.optimizer = optimizer

    def change_loss(self, loss: torch.nn.modules.loss._Loss) -> None:
        """change the loss function

        Parameters
        ----------
        loss : torch.nn.modules.loss._Loss
            loss function
        """

        self.loss = loss


# # write code to test LFDNN class
# if __name__ == "__main__":

#     # create the training data
#     x_train = torch.linspace(-1, 1, 100).reshape(-1, 1)
#     y_train = torch.sin(x_train * 2 * torch.pi)

#     # create the model
#     model = LFDNN(in_features=1,
#                   hidden_features=[20, 20, 20],
#                   out_features=1,
#                   activation="ReLU",
#                   optimizer="Adam",
#                   lr=0.001,
#                   weight_decay=0.01,
#                   loss="mse")

#     # train the model
#     model.train(x_train, y_train, batch_size=100, num_epoch=10000)

#     # create the test data
#     x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
#     y_test = torch.sin(x_test * 2 * torch.pi)

#     # test the model
#     y_pred = model.forward(x_test)
#     import matplotlib.pyplot as plt

#     # plot the results
#     plt.plot(x_test, y_test, label="ground truth")
#     plt.plot(x_test, y_pred.detach().numpy(), label="prediction")
#     plt.legend()
#     plt.show()  # type: ignore
