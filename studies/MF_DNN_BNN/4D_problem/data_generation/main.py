# this script is used to generate data for the 100D problem
import pickle

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

from mfbml.problem_sets.mfb_problems import Meng4D


def get_samples() -> np.ndarray:
    # define function information
    noise_std = 0.05

    # define samples information
    num_hf = 150
    num_lf = 5000
    num_test = 2000

    # initialize the problem
    problem = Meng4D(noise_std=noise_std)

    # in the space of [0,1]
    hf_samples = LatinHypercube(d=4, seed=123).random(n=num_hf)
    lf_samples = LatinHypercube(d=4, seed=456).random(n=num_lf)
    test_samples = LatinHypercube(d=4, seed=789).random(n=num_test)

    # transform the samples to torch tensor
    hf_samples = torch.tensor(hf_samples, dtype=torch.float32)
    lf_samples = torch.tensor(lf_samples, dtype=torch.float32)
    test_samples = torch.tensor(test_samples, dtype=torch.float32)
    # get the function values
    hf_responses = problem.hf(hf_samples, noise_hf=0.01)
    lf_responses = problem.lf(lf_samples, noise_lf=0.05)

    # noiseless test responses
    test_responses_noiseless = problem.hf(test_samples, noise_hf=0.0)
    # noisy test responses
    test_responses_noisy = problem.hf(test_samples, noise_hf=0.01)
    # calculate the noise response for lf
    lf_responses_noisy = problem.lf(test_samples, noise_lf=0.05)
    #  calculate the noiseless response for lf
    lf_responses_noiseless = problem.lf(test_samples, noise_lf=0.0)

    # data dictionary
    data = {
        "hf_samples": hf_samples,
        "lf_samples": lf_samples,
        "test_samples": test_samples,
        "hf_responses": hf_responses,
        "lf_responses": lf_responses,
        "test_responses_noiseless": test_responses_noiseless,
        "test_responses_noisy": test_responses_noisy,
        "lf_responses_noisy": lf_responses_noisy,
        "lf_responses_noiseless": lf_responses_noiseless,

    }
    # save the data to pickle file
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    get_samples()
