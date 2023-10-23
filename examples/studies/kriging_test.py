import time

import matplotlib.pyplot as plt
import numpy as np
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.models.kriging import Kriging
from mfpml.problems.singlefidelity_functions import Hartman6, Sixhump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

func = Sixhump()
num_dim = func.num_dim

samplers = LatinHyperCube(design_space=func.design_space, seed=1)
sample_x = samplers.get_samples(num_samples=10 * num_dim)
sample_y = func.f(sample_x)

model = Kriging(design_space=func.input_domain)
start_time = time.time()
model.train(sample_x=sample_x, sample_y=sample_y)
end_time = time.time()
print('training time: ', end_time - start_time)

# generate test samples
samplers = LatinHyperCube(design_space=func.design_space, seed=2)
test_x = samplers.get_samples(num_samples=100 * num_dim)
test_y = func.f(test_x)

# accuracy test
time_start = time.time()
pred_y, pred_std = model.predict(x_predict=test_x, return_std=True)
time_end = time.time()
print('prediction time: ', time_end - time_start)
mae = mean_absolute_error(test_y, pred_y)
mse = mean_squared_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)

print('mae: ', mae)
print('mse: ', mse)
print('r2: ', r2)
