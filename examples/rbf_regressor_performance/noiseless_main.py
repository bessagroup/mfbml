# this script is used to test the performance of rbf model without noise

import numpy as np
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.problems.singlefidelity_functions import (Branin, Forrester,
                                                     Hartman3, Sixhump)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mfbml.methods.rbf_regressor import RBFSurrogate

# define the problem
func = Hartman3()

sampler = LatinHyperCube(design_space=func.design_space, seed=123)
sample_x = sampler.get_samples(num_samples=20*func.num_dim)
sample_y = func.f(sample_x)
# test samples
sampler = LatinHyperCube(design_space=func.design_space, seed=456)
test_x = sampler.get_samples(num_samples=100*func.num_dim)
test_y = func.f(test_x)
#
model = RBFSurrogate(design_space=func.input_domain)
model.train(sample_x=sample_x, sample_y=sample_y)
#
pred = model.predict(x_predict=test_x)

mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y, pred)
r2 = r2_score(test_y, pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("R2: ", r2)
