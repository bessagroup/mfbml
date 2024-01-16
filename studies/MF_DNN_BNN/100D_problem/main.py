
# create the configuration of the low-fidelity model
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfbml.methods.mf_dnn_bnn import MFDNNBNN
from mfbml.problem_sets.mfb_problems import MFB1


# define function
func = MFB1(noise_std=0.1, num_dim=1, phi=7000)
num_dim = 10

# use multi-fidelity forrester function to test the performance of the MFDNNBNN class
lf_samples = torch.linspace(-1, 1, 401).reshape(-1, 1)
hf_samples = lf_samples[::10]  # sample every 5 points


# generate responses
lf_responses = func.lf(lf_samples)
hf_responses = func.hf(hf_samples)

# create the configuration of the low-fidelity model
lf_configure = {"in_features": 1,
                "hidden_features": [20, 20],
                "out_features": 1,
                "activation": "Tanh",
                "optimizer": "Adam",
                "lr": 0.001,
                "weight_decay": 0.000001,
                "loss": "mse"}
# create the configuration of the high-fidelity model
hf_configure = {"in_features": 1,
                "hidden_features": [50, 50],
                "out_features": 1,
                "activation": "Tanh",
                "lr": 0.001,
                "sigma": 0.1}
# create the MFDNNBNN object
mfdnnbnn = MFDNNBNN(lf_configure=lf_configure,
                    hf_configure=hf_configure,
                    beta_optimize=False,
                    beta_bounds=[-5, 5])


samples = {"lf": lf_samples,
           "hf": hf_samples}

responses = {"lf": lf_responses,
             "hf": hf_responses}

# lf train config
lf_train_config = {"batch_size": None,
                   "num_epochs": 50000,
                   "print_iter": 100}
hf_train_config = {"num_epochs": 35000,
                   "sample_freq": 100,
                   "print_info": True,
                   "burn_in_epochs": 1000}

# train the MFDNNBNN object
mfdnnbnn.train(samples=samples,
               responses=responses,
               lf_train_config=lf_train_config,
               hf_train_config=hf_train_config
               )
# predict the MFDNNBNN object
y, epistemic, total_unc, aleatoric = mfdnnbnn.predict(
    x=torch.linspace(-1, 1, 1000).reshape(-1, 1))
# lf prediction
lf_y = mfdnnbnn.lf_model.forward(
    torch.linspace(-1, 1, 1000).reshape(-1, 1))
# print the prediction
