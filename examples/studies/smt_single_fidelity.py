import time

import matplotlib.pyplot as plt
import numpy as np
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.problems.singlefidelity_functions import Hartman6
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from smt.surrogate_models import KRG

func = Hartman6()
num_dim = func.num_dim

samplers = LatinHyperCube(design_space=func.design_space, seed=1)
sample_x = samplers.get_samples(num_samples=100 * num_dim)
sample_y = func.f(sample_x)

sm = KRG()
sm.set_training_values(sample_x, sample_y)
start_time = time.time()
sm.train()
end_time = time.time()
print('training time: ', end_time - start_time)

# generate test samples
samplers = LatinHyperCube(design_space=func.design_space, seed=2)
test_x = samplers.get_samples(num_samples=100 * num_dim)
test_y = func.f(test_x)

# accuracy test
time_start = time.time()
pred_y = sm.predict_values(test_x)
# estimated variance
pred_sigma2 = sm.predict_variances(test_x)
time_end = time.time()
print('prediction time: ', time_end - time_start)
mae = mean_absolute_error(test_y, pred_y)
mse = mean_squared_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)

print('mae: ', mae)
print('mse: ', mse)
print('r2: ', r2)
