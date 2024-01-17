
# create the configuration of the low-fidelity model
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfbml.methods.mf_dnn_bnn import MFDNNBNN
from mfbml.problem_sets.mfb_problems import Forrester1b, MengCase1

# define function
func = MengCase1(noise_std=0.05)
num_dim = 1

# use multi-fidelity forrester function to test the performance of the MFDNNBNN class
lf_samples = torch.linspace(0, 1, 201).reshape(-1, 1)
hf_samples = lf_samples[::5]  # sample every 5 points


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
                "sigma": 0.05}
# create the MFDNNBNN object
mfdnnbnn = MFDNNBNN(design_space=torch.tile(torch.Tensor([0, 1]), (num_dim, 1)),
                    lf_configure=lf_configure,
                    hf_configure=hf_configure,
                    beta_optimize=False,
                    beta_bounds=[-5, 5])


samples = {"lf": lf_samples,
           "hf": hf_samples}

responses = {"lf": lf_responses,
             "hf": hf_responses}

# lf train config
lf_train_config = {"batch_size": 1000,
                   "num_epochs": 5000,
                   "print_iter": 100,
                   "data_split": True}
hf_train_config = {"num_epochs": 20000,
                   "sample_freq": 100,
                   "print_info": True,
                   "burn_in_epochs": 10000}

# train the MFDNNBNN object
mfdnnbnn.train(samples=samples,
               responses=responses,
               lf_train_config=lf_train_config,
               hf_train_config=hf_train_config
               )
# predict the MFDNNBNN object
y, epistemic, total_unc, aleatoric = mfdnnbnn.predict(
    x=torch.linspace(0, 1, 1000).reshape(-1, 1))
# lf prediction
lf_y = mfdnnbnn.predict_lf(torch.linspace(0, 1, 1000).reshape(-1, 1))
# print the prediction
print(f"aleatoric: {aleatoric}")
print(f"epistemic: {epistemic}")
# plot

plt.figure()
plt.plot(lf_samples, lf_responses, 'x', label="lf")
plt.plot(hf_samples, hf_responses, 'o', label="hf")
# plot lf prediction
plt.plot(torch.linspace(0, 1, 1000).numpy(),
         lf_y.detach().numpy(), label="lf prediction")
plt.plot(torch.linspace(0, 1, 1000).numpy(), y, label="hf prediction")
plt.plot(torch.linspace(0, 1, 1000).numpy(), func.hf(torch.linspace(
    0, 1, 1000).reshape(-1, 1)).detach().numpy(), label="hf ground truth")
plt.fill_between(torch.linspace(0, 1, 1000).numpy(),
                 (y - 2*epistemic).reshape(-1),
                 (y + 2*epistemic).reshape(-1),
                 alpha=0.5,
                 label="total uncertainty")
plt.legend()
plt.savefig("mfdnnbnn.png", bbox_inches='tight', dpi=300)
plt.show()
