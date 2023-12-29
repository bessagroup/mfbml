import time

import matplotlib.pyplot as plt
import numpy as np
from mfpml.design_of_experiment.multifidelity_samplers import MFSobolSequence
from mfpml.models.gaussian_process import GaussianProcess
from mfpml.problems.multifidelity_functions import (mf_Bohachevsky, mf_Booth,
                                                    mf_Borehole, mf_CurrinExp,
                                                    mf_Hartman3, mf_Hartman6,
                                                    mf_Himmelblau, mf_Park91A,
                                                    mf_Park91B, mf_Sixhump)
# import accuracy metrics
from sklearn.metrics import r2_score

from mfbml.get_methods.accuracy_metrics import (log_likelihood_value,
                                                normalized_mae,
                                                normalized_rmse)
from mfbml.methods.mf_rbf_gpr import MFRBFGPR

# define function
func = mf_Hartman6()
# define sampler
sampler = MFSobolSequence(design_space=func.design_space, seed=3)
sample_x = sampler.get_samples(
    num_hf_samples=50*func.num_dim, num_lf_samples=200*func.num_dim)
sample_y = func(sample_x)
#  add noise to low fidelity
sample_y['lf'] = sample_y['lf'] + np.random.normal(
    loc=0, scale=0.1, size=sample_y['lf'].shape)
# add noise to high fidelity
sample_y['hf'] = sample_y['hf'] + np.random.normal(
    loc=0, scale=0.1, size=sample_y['hf'].shape)

# generate test samples
sampler = MFSobolSequence(design_space=func.design_space, seed=2)
test_x = sampler.get_samples(
    num_hf_samples=100*func.num_dim, num_lf_samples=100*func.num_dim)
test_hy = func.hf(test_x["hf"])
test_ly = func.lf(test_x["hf"])
# test hy noise
test_hy_noise = test_hy + np.random.normal(
    loc=0, scale=0.1, size=test_hy.shape)

# single fidelity kriging
time_start = time.time()
kriging_model = GaussianProcess(
    design_space=func.input_domain, optimizer_restart=20)
kriging_model.train(sample_x=sample_x["hf"], sample_y=sample_y["hf"])
pred_sf_y, pred_sf_std = kriging_model.predict(
    x_predict=test_x["hf"], return_std=True)
time_end = time.time()

# calculate accuracy metrics
nrmse_sf = normalized_rmse(test_hy, pred_sf_y)
nmae_sf = normalized_mae(test_hy, pred_sf_y)
r2_sf = r2_score(test_hy, pred_sf_y)
log_likelihood_value_sf = log_likelihood_value(
    test_hy_noise, pred_sf_y, pred_sf_std)

print("### single fidelity ###")
print("nrmse_sf: ", nrmse_sf)
print("nmae_sf: ", nmae_sf)
print("r2_sf: ", r2_sf)
print("learned noise:", kriging_model.noise)
print("learned noise scaled:", kriging_model.noise/kriging_model.y_std)
print("log_likelihood_value of single fidelity: ", log_likelihood_value_sf)
print("time: ", time_end - time_start)

# define kernel
start_time = time.time()
mfrbfkriging_model = MFRBFGPR(
    design_space=func.input_domain, optimizer_restart=20)
mfrbfkriging_model.train(samples=sample_x, responses=sample_y)
pred_y, pred_std = mfrbfkriging_model.predict(
    x_predict=test_x["hf"], return_std=True)

# get prediction of low fidelity
pred_ly = mfrbfkriging_model.predict_lf(test_xl=test_x["hf"])
end_time = time.time()
# print("time: ", end_time - start_time)

# calculate accuracy metrics
print("### multi fidelity  for high fidelity ###")
nrmse = normalized_rmse(test_hy, pred_y)
nmae = normalized_mae(test_hy, pred_y)
r2 = r2_score(test_hy, pred_y)
log_likelihood_value_mf = log_likelihood_value(
    test_hy_noise, pred_y, pred_std)
cpu_time = end_time - start_time

print("nrmse: ", nrmse)
print("nmae: ", nmae)
print("r2: ", r2)
print("learned noise:", mfrbfkriging_model.noise)
print("learned noise scaled:", mfrbfkriging_model.noise/mfrbfkriging_model.yh_std)
print("log_likelihood_value of multi fidelity: ", log_likelihood_value_mf)
print("cpu_time: ", cpu_time)

# calculate accuracy metrics for low fidelity
nrmse_lf = normalized_rmse(test_ly, pred_ly)
nmae_lf = normalized_mae(test_ly, pred_ly)
r2_lf = r2_score(test_ly, pred_ly)

print("### multi fidelity for low fidelity ###")
print("nrmse_lf: ", nrmse_lf)
print("nmae_lf: ", nmae_lf)
print("r2_lf: ", r2_lf)

# plot std
fig, ax = plt.subplots()
ax.plot(pred_std, label="std")
ax.plot(pred_sf_std, label="std_sf")
ax.legend()
plt.show()
