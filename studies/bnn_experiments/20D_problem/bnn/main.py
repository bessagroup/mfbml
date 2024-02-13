# this script is used to train the BNN model for 4D problem

import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from mfbml.methods.bnn import BNNWrapper
from mfbml.metrics.accuracy_metrics import (log_likelihood_value,
                                            normalized_mae, normalized_rmse)

# in this script, the high-fidelity bnn model is trained using the HF data
# Because the BNNWrapper will not scale data, so the data needs to be scaled.


def normalize_inputs(x: torch.Tensor,
                     design_space: torch.Tensor) -> torch.Tensor:
    """normalize the inputs

    Parameters
    ----------
    x : torch.Tensor
        original inputs
    design_space : torch.Tensor
        design space

    Returns
    -------
    torch.Tensor
        normalized inputs
    """
    x = (x - design_space[:, 0]) / \
        (design_space[:, 1] - design_space[:, 0])

    return x


def normalize_outputs(y: torch.Tensor) -> torch.Tensor:
    """normalize the outputs

    Parameters
    ----------
    y : torch.Tensor
        original outputs

    Returns
    -------
    torch.Tensor
        normalized outputs
    """
    y = (y - torch.mean(y)) / torch.std(y)
    return y


def main() -> None:

    # read data from ../data_generation/data.pkl
    data = pickle.load(open("../data_generation/data_20D_example.pkl", "rb"))
    print(f"HF samples: {data['hf_samples'].shape}")
    print(f"LF samples: {data['lf_samples'].shape}")

    # get the data
    hf_samples = data["hf_samples"]
    test_samples = data["test_samples"]
    hf_responses = data["hf_responses"]
    test_responses_noiseless = data["test_responses_noiseless"]
    test_responses_noisy = data["test_responses_noisy"]

    # design space
    design_space = torch.tile(torch.Tensor([-3, 3]), (hf_samples.shape[1], 1))

    # scale the data
    hf_samples_scaled = normalize_inputs(hf_samples, design_space)
    test_samples_scaled = normalize_inputs(test_samples, design_space)
    hf_responses_scaled = normalize_outputs(hf_responses)

    # scale the sigma
    sigma_scaled = float(10.0 / torch.std(hf_responses))
    # define the bnn model
    bnn_model = BNNWrapper(in_features=20,
                           hidden_features=[512, 512],
                           out_features=1,
                           activation="ReLU",
                           lr=0.001,
                           sigma=sigma_scaled)

    # train the bnn model
    bnn_model.train(x=hf_samples_scaled,
                    y=hf_responses_scaled,
                    num_epochs=50000,
                    sample_freq=100,
                    burn_in_epochs=10000,
                    print_info=True)

    # predict the MFDNNBNN object
    y, epistemic, total_unc, aleatoric = bnn_model.predict(
        x=test_samples_scaled)

    # scale the data back
    y = y * torch.std(hf_responses).numpy() + torch.mean(hf_responses).numpy()
    total_unc = total_unc * torch.std(hf_responses).numpy()
    aleatoric = aleatoric * torch.std(hf_responses).numpy()
    epistemic = epistemic * torch.std(hf_responses).numpy()

    # calculate the nmae, nrmse, r2 score and log likelihood
    nmae = normalized_mae(test_responses_noiseless.numpy(), y)
    nrmse = normalized_rmse(test_responses_noiseless.numpy(), y)
    r2 = r2_score(test_responses_noiseless.numpy(), y)

    # calculate the log likelihood
    log_likelihood = log_likelihood_value(
        test_responses_noisy.numpy(), y, total_unc)

    # save the results
    results = {"nmae": nmae,
               "nrmse": nrmse,
               "r2": r2,
               "log_likelihood": log_likelihood}

    # save the results to csv file
    df = pd.DataFrame(results, index=[0])
    df.to_csv("bnn_20D_problem.csv", index=False)


if __name__ == "__main__":
    main()
