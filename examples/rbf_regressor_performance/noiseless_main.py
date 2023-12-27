# this script is used to test the performance of rbf model without noise

import numpy as np
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.problems.singlefidelity_functions import (Branin, Forrester,
                                                     Hartman3, Hartman6,
                                                     Sixhump)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mfbml.methods.rbf_regressor import RBFSurrogate

# define the problem
func = Hartman6()

sampler = LatinHyperCube(design_space=func.design_space, seed=123)
sample_x = sampler.get_samples(num_samples=10*func.num_dim)
sample_y = func.f(sample_x)
# test samples
sampler = LatinHyperCube(design_space=func.design_space, seed=456)
test_x = sampler.get_samples(num_samples=100*func.num_dim)
test_y = func.f(test_x)
#
model = RBFSurrogate(design_space=func.input_domain,
                     params_optimize=True, optimizer_restart=5)
model.train(sample_x=sample_x, sample_y=sample_y, portion_test=0.2)
#
pred = model.predict(x_predict=test_x)

mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y, pred)
r2 = r2_score(test_y, pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("R2: ", r2)

mae_train = mean_absolute_error(sample_y, model.predict(x_predict=sample_x))
mse_train = mean_squared_error(sample_y, model.predict(x_predict=sample_x))
r2_train = r2_score(sample_y, model.predict(x_predict=sample_x))
print("MAE_train: ", mae_train)
print("MSE_train: ", mse_train)
print("R2_train: ", r2_train)

print("params: ", model.kernel.param)
