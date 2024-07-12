# create the configuration of the low-fidelity model
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from mfbml.problems.high_dimension_problems import MengCase1


def main():
    # define function
    func = MengCase1(noise_std=0.0)
    # generate samples (21 HF samples, 201 LF samples)
    lf_samples = torch.linspace(0, 1, 201).reshape(-1, 1)
    hf_samples = torch.linspace(0.05, 0.95, 11).reshape(-1, 1)
    # add the end points

    # training configure
    samples = {"lf": lf_samples, "hf": hf_samples}

    # generate responses
    lf1_responses = func.lf1(lf_samples, noise_lf=0.05)
    lf2_responses = func.lf2(lf_samples, noise_lf=0.05)
    lf3_responses = func.lf3(lf_samples, noise_lf=0.05)
    # get the high-fidelity responses
    hf_responses = func.hf(hf_samples, noise_hf=0.05)
    # generate the test points
    test_samples = torch.linspace(0, 1, 1001).reshape(-1, 1)
    # noiseless responses
    test_hf_responses_noiseless = func.hf(test_samples, noise_hf=0.0)
    test_lf1_responses_noiseless = func.lf1(test_samples, noise_lf=0.0)
    test_lf2_responses_noiseless = func.lf2(test_samples, noise_lf=0.0)
    test_lf3_responses_noiseless = func.lf3(test_samples, noise_lf=0.0)
    # noise responses
    test_hf_responses = func.hf(test_samples, noise_hf=0.05)

    # dataset of lf1 and hf
    responses_lf1 = {"lf": lf1_responses,
                     "hf": hf_responses}
    # dataset of lf2 and hf
    responses_lf2 = {"lf": lf2_responses,
                     "hf": hf_responses}
    # dataset of lf2 and hf
    responses_lf3 = {"lf": lf3_responses,
                     "hf": hf_responses}

    # plot the function
    fig, ax = plt.subplots()
    ax.plot(test_samples, test_hf_responses_noiseless, label="HF")
    ax.plot(test_samples, test_lf1_responses_noiseless,   label="LF1")
    ax.plot(test_samples, test_lf2_responses_noiseless,  label="LF2")
    ax.plot(test_samples, test_lf3_responses_noiseless,  label="LF3")

    # plot the samples
    ax.scatter(hf_samples, hf_responses, label="HF samples")
    ax.scatter(lf_samples, lf1_responses, alpha=0.5, label="LF1 samples")
    ax.scatter(lf_samples, lf2_responses, alpha=0.5,  label="LF2 samples")
    ax.scatter(lf_samples, lf3_responses, alpha=0.5, label="LF3 samples")

    plt.legend()
    # save the figure
    plt.savefig("sample_plan.png", dpi=300, bbox_inches="tight")
    plt.close()

    # correlation between HF and LF samples
    print("Correlation between HF and LF samples")
    print(pearsonr(test_hf_responses_noiseless.flatten(),
                   test_lf1_responses_noiseless.flatten()))
    print(pearsonr(test_hf_responses_noiseless.flatten(),
                   test_lf2_responses_noiseless.flatten()))
    print(pearsonr(test_hf_responses_noiseless.flatten(),
                   test_lf3_responses_noiseless.flatten()))

    # save the data
    dict_data = {"samples": samples,
                 "responses_lf1": responses_lf1,
                 "responses_lf2": responses_lf2,
                 "responses_lf3": responses_lf3,
                 "test_samples": test_samples,
                 "test_hf_responses": test_hf_responses,
                 "test_hf_responses_noiseless": test_hf_responses_noiseless,
                 "test_lf1_responses": test_lf1_responses_noiseless,
                 "test_lf2_responses": test_lf2_responses_noiseless,
                 "test_lf3_responses": test_lf3_responses_noiseless,
                 "correlation": [pearsonr(test_hf_responses_noiseless.flatten(),
                                          test_lf1_responses_noiseless.flatten()),
                                 pearsonr(test_hf_responses_noiseless.flatten(),
                                          test_lf2_responses_noiseless.flatten()),
                                 pearsonr(test_hf_responses_noiseless.flatten(),
                                          test_lf3_responses_noiseless.flatten())]}
    # save to pickle
    with open("data.pkl", "wb") as f:
        pickle.dump(dict_data, f)


if __name__ == "__main__":
    main()
