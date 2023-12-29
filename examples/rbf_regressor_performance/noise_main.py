# this script is used to test the performance of RBF regressor with noise data
# this script is used to test the performance of rbf model without noise

import numpy as np
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.models.gaussian_process import GaussianProcess
from mfpml.problems.singlefidelity_functions import (Branin, Forrester,
                                                     Hartman3, Hartman6,
                                                     Sixhump)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mfbml.methods.rbf_regressor import RBFKernelRegression

# define the problem
func = Branin()

sampler = LatinHyperCube(design_space=func.design_space, seed=123)
sample_x = sampler.get_samples(num_samples=10*func.num_dim)
sample_y = func.f(sample_x)
# test samples
sampler = LatinHyperCube(design_space=func.design_space, seed=456)
test_x = sampler.get_samples(num_samples=100*func.num_dim)
test_y = func.f(test_x) + np.random.normal(0, 0.1, size=(test_x.shape[0], 1))
#
model = RBFKernelRegression(design_space=func.input_domain,
                            params_optimize=True,
                            noise_data=True,
                            optimizer_restart=5)
model.train(sample_x=sample_x, sample_y=sample_y, portion_test=0.2)
#
pred = model.predict(x_predict=test_x)

mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y, pred)
r2 = r2_score(test_y, pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("R2: ", r2)
print("noise rbf: ", model.noise_std)
print("params: ", model.kernel.param)
mae_train = mean_absolute_error(sample_y, model.predict(x_predict=sample_x))
mse_train = mean_squared_error(sample_y, model.predict(x_predict=sample_x))
r2_train = r2_score(sample_y, model.predict(x_predict=sample_x))
print("MAE_train: ", mae_train)
print("MSE_train: ", mse_train)
print("R2_train: ", r2_train)


# kriging model
kriging_model = GaussianProcess(design_space=func.input_domain,
                                optimizer_restart=5)
kriging_model.train(sample_x=sample_x, sample_y=sample_y)
pred = kriging_model.predict(x_predict=test_x)

# accuracy
mae_kriging = mean_absolute_error(test_y, pred)
mse_kriging = mean_squared_error(test_y, pred)
r2_kriging = r2_score(test_y, pred)
# print
print("MAE_kriging: ", mae_kriging)
print("MSE_kriging: ", mse_kriging)
print("R2_kriging: ", r2_kriging)
