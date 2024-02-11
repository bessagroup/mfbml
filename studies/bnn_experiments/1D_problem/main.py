# create the configuration of the low-fidelity model
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
# get the accuracy metrics
from sklearn.metrics import r2_score

from mfbml.methods.bnn import BNNWrapper
from mfbml.methods.mf_dnn_bnn import MFDNNBNN
from mfbml.methods.sequential_mf_bnn import SequentialMFBNN
from mfbml.metrics import (mean_log_likelihood_value, normalized_mae,
                           normalized_rmse)
from mfbml.problems.high_dimension_problems import MengCase1

# ignore the warnings
warnings.filterwarnings("ignore")


def single_run(iteration: int) -> dict[str, Any]:
    print("============== Iteration: ", iteration, " ==============")

    # define function
    func = MengCase1(noise_std=0.05)
    # generate samples uniformly
    lf_samples = torch.linspace(0, 1, 201).reshape(-1, 1)
    hf_samples = torch.linspace(0, 1, 21).reshape(-1, 1)

    # generate responses
    lf_responses = func.lf(lf_samples)
    hf_responses = func.hf(hf_samples)

    # generate the test points
    test_samples = torch.linspace(0, 1, 1001).reshape(-1, 1)
    # noiseless responses
    test_hf_responses_noiseless = func.hf(test_samples, noise_hf=0.0)
    test_hf_responses = func.hf(test_samples)

    # create the configuration of the low-fidelity model
    lf_configure = {
        "in_features": 1,
        "hidden_features": [20, 20],
        "out_features": 1,
        "activation": "Tanh",
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 0.000001,
        "loss": "mse",
    }
    # create the configuration of the high-fidelity model
    hf_parallel_configure = {
        "in_features": 1,
        "hidden_features": [50, 50],
        "out_features": 1,
        "activation": "Tanh",
        "lr": 0.001,
        "sigma": 0.05,
    }
    #
    hf_sequential_configure = {
        "in_features": 2,
        "hidden_features": [50, 50],
        "out_features": 1,
        "activation": "Tanh",
        "lr": 0.001,
        "sigma": 0.05,
    }

    # training configure
    samples = {"lf": lf_samples, "hf": hf_samples}
    responses = {"lf": lf_responses, "hf": hf_responses}

    # lf train config
    lf_train_config = {
        "batch_size": None,
        "num_epochs": 10000,
        "print_iter": 1000,
        "data_split": False,
    }
    hf_train_config = {
        "num_epochs": 20000,
        "sample_freq": 100,
        "print_info": False,
        "burn_in_epochs": 10000,
    }

    # ================================   BNN   ================================== #
    # define BNN
    bnn_model = BNNWrapper(
        in_features=1,
        hidden_features=[50, 50],
        out_features=1,
        activation="Tanh",
        lr=0.001,
        sigma=0.05,
    )

    # train the model
    bnn_model.train(
        x=hf_samples,
        y=hf_responses,
        num_epochs=20000,
        sample_freq=100,
        burn_in_epochs=10000,
        print_info=False,
    )
    # visualize the posterior of bnn
    bnn_y, bnn_epistemic, bnn_total_unc, bnn_aleatoric = bnn_model.predict(
        x=test_samples)
    print("Finished training standard BNN")
    # ========================   Sequential MF-BNN   ======================== #
    sequential_bnn = SequentialMFBNN(
        design_space=torch.Tensor([[0, 1]]),
        lf_configure=lf_configure,
        hf_configure=hf_sequential_configure,
    )
    sequential_bnn.train(
        samples=samples,
        responses=responses,
        lf_train_config=lf_train_config,
        hf_train_config=hf_train_config,
    )
    (
        sequential_bnn_y,
        sequential_bnn_epistemic,
        sequential_bnn_total_unc,
        sequential_bnn_aleatoric,
    ) = sequential_bnn.predict(x=test_samples)

    print("Finished training sequential MF-BNN")

    # ============================   MFDNNBNN   ================================= #
    mfdnnbnn = MFDNNBNN(
        design_space=torch.Tensor([[0, 1]]),
        lf_configure=lf_configure,
        hf_configure=hf_parallel_configure,
        beta_optimize=False,
        beta_bounds=[-5, 5],
    )
    # define beta
    mfdnnbnn.beta = np.array([2.0])
    mfdnnbnn.train(
        samples=samples,
        responses=responses,
        lf_train_config=lf_train_config,
        hf_train_config=hf_train_config,
    )

    # predict the MFDNNBNN object
    (
        y_proposed,
        epistemic_proposed,
        total_unc_proposed,
        aleatoric_proposed,
    ) = mfdnnbnn.predict(x=test_samples)
    print("Finished training MF-DNN-BNN")

    # ============================   Metrics   ============================== #
    nrmse_bnn = normalized_rmse(test_hf_responses_noiseless.numpy(), bnn_y)
    nmae_bnn = normalized_mae(test_hf_responses_noiseless.numpy(), bnn_y)
    r2_bnn = r2_score(test_hf_responses_noiseless.numpy(), bnn_y)
    ll_bnn = mean_log_likelihood_value(
        test_hf_responses.numpy(), bnn_y, bnn_total_unc)

    # sequential mf-bnn
    nrmse_smfbnn = normalized_rmse(
        test_hf_responses_noiseless.numpy(), sequential_bnn_y
    )
    nmae_smfbnn = normalized_mae(
        test_hf_responses_noiseless.numpy(), sequential_bnn_y
    )
    ll_smfbnn = mean_log_likelihood_value(
        test_hf_responses.numpy(),
        sequential_bnn_y,
        sequential_bnn_total_unc,
    )
    r2_smfbnn = r2_score(test_hf_responses_noiseless.numpy(), sequential_bnn_y)

    nrmse_mfdnnbnn = normalized_rmse(
        test_hf_responses_noiseless.numpy(), y_proposed)
    nmae_mfdnnbnn = normalized_mae(
        test_hf_responses_noiseless.numpy(), y_proposed)
    ll_mfdnnbnn = mean_log_likelihood_value(
        test_hf_responses.numpy(), y_proposed, total_unc_proposed
    )
    r2_mfdnnbnn = r2_score(test_hf_responses_noiseless.numpy(), y_proposed)

    results = {
        "nrmse_bnn": nrmse_bnn,
        "nmae_bnn": nmae_bnn,
        "r2_bnn": r2_bnn,
        "ll_bnn": ll_bnn,
        "nrmse_smfbnn": nrmse_smfbnn,
        "nmae_smfbnn": nmae_smfbnn,
        "r2_smfbnn": r2_smfbnn,
        "ll_smfbnn": ll_smfbnn,
        "nrmse_mfdnnbnn": nrmse_mfdnnbnn,
        "nmae_mfdnnbnn": nmae_mfdnnbnn,
        "r2_mfdnnbnn": r2_mfdnnbnn,
        "ll_mfdnnbnn": ll_mfdnnbnn,
    }

    # print the results
    print("BNN: ", results["nrmse_bnn"], results["nmae_bnn"],
          results["r2_bnn"], results["ll_bnn"])
    print(
        "Sequential MF-BNN: ",
        results["nrmse_smfbnn"],
        results["nmae_smfbnn"],
        results["r2_smfbnn"],
        results["ll_smfbnn"],
    )
    print(
        "MF-DNN-BNN: ",
        results["nrmse_mfdnnbnn"],
        results["nmae_mfdnnbnn"],
        results["r2_mfdnnbnn"],
        results["ll_mfdnnbnn"],
    )

    # plot the results
    font_size = 12
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # plot results of bnn
    ax[0].plot(
        samples["hf"].numpy(),
        responses["hf"].numpy(),
        "kx",
        linewidth=2,
        markersize=10,
        label="HF samples",
    )
    ax[0].plot(
        test_samples.numpy(),
        test_hf_responses_noiseless.numpy(),
        "--",
        color="#33BBEE",
        linewidth=2,
        label="HF truth",
    )
    ax[0].plot(
        test_samples.numpy(),
        bnn_y,
        "-",
        color="#CC3311",
        linewidth=2,
        label="HF prediction",
    )
    ax[0].fill_between(
        test_samples.flatten(),
        (bnn_y - 2 * bnn_total_unc).flatten(),
        (bnn_y + 2 * bnn_total_unc).flatten(),
        alpha=0.5,
        color="#BBBBBB",
        label="CI interval",
    )
    ax[0].legend(loc="lower right")
    ax[0].set_title("BNN", fontsize=font_size)
    ax[0].set_xlabel("x", fontsize=font_size)
    ax[0].set_ylabel("y", fontsize=font_size)
    ax[0].tick_params(labelsize=font_size)
    ax[0].set_ylim([-2, 1.0])

    #  plot for sequential mf-bnn
    ax[1].plot(
        samples["hf"].numpy(),
        responses["hf"].numpy(),
        "kx",
        linewidth=2,
        markersize=10,
        label="HF samples",
    )
    ax[1].plot(
        test_samples.numpy(),
        test_hf_responses_noiseless.numpy(),
        "--",
        color="#33BBEE",
        linewidth=2,
        label="HF noiseless truth",
    )
    ax[1].plot(
        test_samples.numpy(),
        sequential_bnn_y,
        "-",
        color="#CC3311",
        linewidth=2,
        label="HF prediction",
    )

    ax[1].fill_between(
        test_samples.flatten().numpy(),
        (sequential_bnn_y - 2 * sequential_bnn_total_unc).flatten(),
        (sequential_bnn_y + 2 * sequential_bnn_total_unc).flatten(),
        alpha=0.5,
        color="#BBBBBB",
        label="CI interval",
    )
    ax[1].legend(loc="lower right")
    ax[1].set_title("Meng's MF-BNN", fontsize=font_size)
    ax[1].set_xlabel("x", fontsize=font_size)
    ax[1].set_ylabel("y", fontsize=font_size)
    ax[1].tick_params(labelsize=font_size)
    ax[1].set_ylim([-2, 1.0])

    # plot for MFDNNBNN
    ax[2].plot(
        samples["hf"].numpy(),
        responses["hf"].numpy(),
        "kx",
        linewidth=2,
        markersize=10,
        label="HF samples",
    )
    ax[2].plot(
        test_samples.numpy(),
        test_hf_responses_noiseless.numpy(),
        "--",
        color="#33BBEE",
        linewidth=2,
        label="HF truth",
    )
    ax[2].plot(
        test_samples.numpy(),
        y_proposed,
        "-",
        color="#CC3311",
        linewidth=2,
        label="HF prediction",
    )

    ax[2].fill_between(
        test_samples.flatten().numpy(),
        (y_proposed - 2 * total_unc_proposed).flatten(),
        (y_proposed + 2 * total_unc_proposed).flatten(),
        alpha=0.5,
        color="#BBBBBB",
        label="CI interval",
    )

    ax[2].set_title("MF-DNN-BNN", fontsize=font_size)
    ax[2].set_xlabel("x", fontsize=font_size)
    ax[2].set_ylabel("y", fontsize=font_size)
    ax[2].tick_params(labelsize=font_size)
    ax[2].set_ylim([-2, 1.0])
    ax[2].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"mf_dnn_bnn_known_noise_{iteration}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(
        f"mf_dnn_bnn_known_noise_{iteration}.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        f"mf_dnn_bnn_known_noise_{iteration}.svg", dpi=300, bbox_inches="tight")

    return results


# run the experiment
# fix the random seed for reproducibility
np.random.seed(2)
torch.manual_seed(2)
results = single_run(1)
