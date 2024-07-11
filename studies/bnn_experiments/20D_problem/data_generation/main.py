# this script is used to generate data for the 100D problem
import pickle

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

from mfbml.problems.high_dimension_problems import Meng20D


def scale_samples(samples: np.ndarray, design_space: np.ndarray) -> np.ndarray:
    """
    Scale the samples to the design space
    """
    # get samples information
    num_samples = samples.shape[0]
    num_dim = samples.shape[1]
    # initialize the scaled samples
    scaled_samples = np.zeros((num_samples, num_dim))
    # scale the samples to the design space
    for i in range(design_space.shape[0]):
        scaled_samples[:, i] = samples[:, i] * \
            (design_space[i, 1]-design_space[i, 0]) + design_space[i, 0]
    return scaled_samples


def get_samples(num_dim: int) -> np.ndarray:
    # define function information
    noise_std = 10.0

    # define samples information
    num_hf = 100*num_dim
    num_lf = 1000*num_dim
    num_test = 1000*num_dim

    # initialize the problem
    problem = Meng20D(num_dim=num_dim, noise_std=noise_std)

    # define the design space which [-1,1]
    design_space = np.tile(np.array([-3, 3]), (num_dim, 1))
    # in the space of [0,1]
    hf_samples = LatinHypercube(d=num_dim, seed=123).random(n=num_hf)
    lf_samples = LatinHypercube(d=num_dim, seed=456).random(n=num_lf)
    test_samples = LatinHypercube(d=num_dim, seed=789).random(n=num_test)
    # scale the samples to the design space
    hf_samples = scale_samples(hf_samples, design_space)
    lf_samples = scale_samples(lf_samples, design_space)
    test_samples = scale_samples(test_samples, design_space)

    # transform the samples to torch tensor
    hf_samples = torch.tensor(hf_samples, dtype=torch.float32)
    lf_samples = torch.tensor(lf_samples, dtype=torch.float32)
    test_samples = torch.tensor(test_samples, dtype=torch.float32)
    # get the function values
    hf_responses = problem.hf(hf_samples)
    lf_responses = problem.lf(lf_samples, noise_lf=0.0)

    # noiseless test responses
    test_responses_noiseless = problem.hf(test_samples, noise_hf=0.0)
    # noisy test responses
    test_responses_noisy = problem.hf(test_samples)
    # calculate the noise response for lf
    lf_responses_noisy = problem.lf(test_samples)
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
    with open("data_20D_example.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    get_samples(num_dim=20)
