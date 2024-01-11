
# create the configuration of the low-fidelity model
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfbml.get_methods.accuracy_metrics import (log_likelihood_value,
                                                normalized_mae,
                                                normalized_rmse)
from mfbml.methods.mf_dnn_bnn import MFDNNBNN
from mfbml.problem_sets.torch_problems import MengCase1


def function_fit(beta: float) -> float:
    # define function
    func = MengCase1(noise_std=0.05)

    # generating samples for training
    lf_samples = torch.linspace(0, 1, 201).reshape(-1, 1)
    hf_samples = lf_samples[::10]  # sample every 5 points
    # generate responses
    lf_responses = func.lf(lf_samples)
    hf_responses = func.hf(hf_samples)
    # dictionary of samples and responses
    samples = {"lf": lf_samples,
               "hf": hf_samples}
    responses = {"lf": lf_responses,
                 "hf": hf_responses}

    # generating test samples
    x_test = torch.linspace(0, 1, 1000).reshape(-1, 1)
    y_test_noiseless = func.hf(x_test, noise_hf=0)
    y_test = func.hf(x_test)

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
    mfdnnbnn = MFDNNBNN(lf_configure=lf_configure,
                        hf_configure=hf_configure,
                        beta_optimize=False,
                        beta_bounds=[-5, 5])
    # define beta
    mfdnnbnn.beta = np.array([beta])
    # lf train config
    lf_train_config = {"batch_size": None,
                       "num_epochs": 50000,
                       "print_iter": 100}
    hf_train_config = {"num_epochs": 50000,
                       "sample_freq": 100,
                       "print_info": True,
                       "burn_in_epochs": 30000}

    # train the MFDNNBNN object
    mfdnnbnn.train(samples=samples,
                   responses=responses,
                   lf_train_config=lf_train_config,
                   hf_train_config=hf_train_config
                   )
    # predict the MFDNNBNN object
    y, epistemic, total_unc, aleatoric = mfdnnbnn.predict(
        x=x_test)

    # get error metrics for test data
    nrmse = normalized_rmse(y_test_noiseless.numpy(), y)
    nmae = normalized_mae(y_test_noiseless.numpy(), y)
    log_likelihood = log_likelihood_value(y_test.numpy(), y, total_unc)

    # get error metrics for training data
    y_train, epistemic_train, total_unc_train, aleatoric_train = \
        mfdnnbnn.predict(x=hf_samples)
    log_likelihood_train = log_likelihood_value(
        hf_responses.numpy(), y_train, total_unc_train)

    return nrmse, nmae, log_likelihood, log_likelihood_train


def main():
    # define beta
    beta_list = np.linspace(-4, 4, 81)
    # create a pandas dataframe to store the results which are
    # nrmse, nmae, log_likelihood, log_likelihood_train
    results = np.zeros((len(beta_list), 5))
    for i, beta in enumerate(beta_list):
        # get the results
        results[i, 0] = beta
        results[i, 1:] = function_fit(beta)
        # save the results with header
        np.savetxt("results.csv", results, delimiter=",",
                   header="beta, nrmse, nmae, log_likelihood, log_likelihood_train")


if __name__ == "__main__":
    main()
