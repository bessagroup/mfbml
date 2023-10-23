import time

import numpy as np
from mfpml.design_of_experiment.multifidelity_samplers import MFLatinHyperCube
from mfpml.optimization.evolutionary_algorithms import DE
from mfpml.problems.multifidelity_functions import (mf_Hartman3, mf_Hartman6,
                                                    mf_Sixhump)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rbfgp import RBFboostedGP

# define function
func = mf_Hartman3()
num_dim = func.num_dim

# define sampler
sampler = MFLatinHyperCube(design_space=func.design_space, seed=1)
sample_x = sampler.get_samples(
    num_hf_samples=10 * num_dim, num_lf_samples=100 * num_dim)
sample_y = func(sample_x)

# generate test samples
sampler = MFLatinHyperCube(design_space=func.design_space, seed=2)
test_x = sampler.get_samples(
    num_hf_samples=100 * num_dim, num_lf_samples=100 * num_dim)

# define kernel
model = RBFboostedGP(design_space=func.input_domain)
model.train(samples=sample_x, responses=sample_y)
start_time = time.time()
pred_y, pred_std = model.predict(x_predict=test_x['hf'], return_std=True)
end_time = time.time()
print('prediction time: ', end_time - start_time)

# accuracy test
mae = mean_absolute_error(func.hf(test_x['hf']), pred_y)
mse = mean_squared_error(func.hf(test_x['hf']), pred_y)
r2 = r2_score(func.hf(test_x['hf']), pred_y)
print('mae: ', mae)
print('mse: ', mse)
print('r2: ', r2)
