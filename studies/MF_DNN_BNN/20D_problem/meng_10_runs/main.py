
import pickle

import pandas as pd
import torch
from sklearn.metrics import r2_score

from mfbml.get_methods.accuracy_metrics import (log_likelihood_value,
                                                normalized_mae,
                                                normalized_rmse)
from mfbml.methods.sequential_mf_bnn import SequentialMFBNN


def single_run(iter: int) -> dict:

    print(f"iter: {iter}")

    # read data from ../data_generation/data.pkl
    data = pickle.load(open("../data_generation/data_20D_example.pkl", "rb"))
    print(data["hf_samples"].shape)
    # get the data
    hf_samples = data["hf_samples"]
    lf_samples = data["lf_samples"]
    test_samples = data["test_samples"]
    hf_responses = data["hf_responses"]
    lf_responses = data["lf_responses"]
    test_responses_noiseless = data["test_responses_noiseless"]
    test_responses_noisy = data["test_responses_noisy"]
    lf_responses_noisy = data["lf_responses_noisy"]

    # design space
    design_space = torch.tile(torch.Tensor([-3, 3]), (hf_samples.shape[1], 1))

    # create the samples and responses dictionary
    samples = {"lf": lf_samples,
               "hf": hf_samples}

    responses = {"lf": lf_responses,
                 "hf": hf_responses}

    # create the configuration of the low-fidelity model
    lf_configure = {"in_features": 20,
                    "hidden_features": [200, 200],
                    "out_features": 1,
                    "activation": "Tanh",
                    "optimizer": "Adam",
                    "lr": 0.0001,
                    "weight_decay": 0.00003,
                    "loss": "mse"}

    # create the configuration of the high-fidelity model
    hf_configure = {"in_features": 21,
                    "hidden_features": [512, 512],
                    "out_features": 1,
                    "activation": "ReLU",
                    "lr": 0.001,
                    "sigma": 10.0}

    # create the sequential mf bnn object
    mfbnn = SequentialMFBNN(design_space=design_space,
                            lf_configure=lf_configure,
                            hf_configure=hf_configure)

    # lf train config
    lf_train_config = {"batch_size": 1000,
                       "num_epochs": 80000,
                       "print_iter": 100,
                       "data_split": True}
    hf_train_config = {"num_epochs": 50000,
                       "sample_freq": 100,
                       "print_info": True,
                       "burn_in_epochs": 10000}

    # train the MFDNNBNN object
    mfbnn.train(samples=samples,
                responses=responses,
                lf_train_config=lf_train_config,
                hf_train_config=hf_train_config
                )

    # predict the MFDNNBNN object
    y, epistemic, total_unc, aleatoric = mfbnn.predict(x=test_samples)
    print(f"aleatoric: {aleatoric}")
    print(f"epistemic: {epistemic}")
    print(f"total_unc: {total_unc}")
    # lf prediction
    lf_y = mfbnn.predict_lf(x=test_samples, output_format="numpy")

    # calculate the nmae, nrmse, r2 score and log likelihood
    nmae = normalized_mae(test_responses_noiseless.numpy(), y)
    nrmse = normalized_rmse(test_responses_noiseless.numpy(), y)
    r2 = r2_score(test_responses_noiseless.numpy(), y)

    # calculate the log likelihood
    log_likelihood = log_likelihood_value(
        test_responses_noisy.numpy(), y, total_unc)

    # lf nmae, nrmse, r2 score
    lf_nmae = normalized_mae(lf_responses_noisy.numpy(), lf_y)
    lf_nrmse = normalized_rmse(
        lf_responses_noisy.numpy(), lf_y)
    lf_r2 = r2_score(lf_responses_noisy.numpy(), lf_y)

    # save the results
    results = {"nmae": nmae,
               "nrmse": nrmse,
               "r2": r2,
               "log_likelihood": log_likelihood,
               "lf_nmae": lf_nmae,
               "lf_nrmse": lf_nrmse,
               "lf_r2": lf_r2}

    # save the results to csv file
    df = pd.DataFrame(results, index=[0])
    df.to_csv(f"meng_mf_bnn_20D_results_{iter}.csv", index=False)

    return results


def main() -> None:
    # create a pandas dataframe to store the results
    results = pd.DataFrame(columns=["nmae",
                                    "nrmse",
                                    "r2",
                                    "log_likelihood",
                                    "lf_nmae",
                                    "lf_nrmse",
                                    "lf_r2"])

    for iter in range(10):
        result = single_run(iter)
        # save the result to the dataframe
        results.loc[iter] = result
        results.to_csv("meng_mf_bnn_20D_results.csv", index=False)


if __name__ == "__main__":
    main()
