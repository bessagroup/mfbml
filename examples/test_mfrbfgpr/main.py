import time

import matplotlib.pyplot as plt
import numpy as np
from mfpml.design_of_experiment.multifidelity_samplers import MFSobolSequence
from mfpml.models.gaussian_process import GaussianProcess
from mfpml.models.kernels import RBF
from mfpml.optimization.evolutionary_algorithms import DE
from mfpml.problems.multifidelity_functions import Forrester_1b
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize
# import accuracy measures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mfbml.methods.mfrbfgp import MFRBFGPR

func = Forrester_1b()

noise = 0.3
# define sampler
sampler = MFSobolSequence(design_space=func.design_space, seed=0)
sample_x = sampler.get_samples(num_hf_samples=50, num_lf_samples=200)
# update sample_x['hf']
# sample_x['hf'] = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))
sample_y = {}
# get response
sample_y["hf"] = func.hf(sample_x["hf"]) + np.random.normal(
    0, noise, size=sample_x["hf"].shape[0]
).reshape(-1, 1)
sample_y["lf"] = func.lf(sample_x["lf"]) + np.random.normal(
    0, noise, size=sample_x["lf"].shape[0]
).reshape(-1, 1)

# generate test samples
test_x = np.linspace(0, 1, 1000).reshape(-1, 1)
test_hy = func.hf(test_x)
test_ly = func.lf(test_x)

# generate noisy test data test_x_noisy
test_x_noisy = np.linspace(0, 1, 1000).reshape(-1, 1)
test_hy_noisy = func.hf(test_x_noisy) + np.random.normal(
    0, noise, size=test_x_noisy.shape[0]
).reshape(-1, 1)


# define kernel
# start_time = time.time()
mfrbfgp = MFRBFGPR(
    design_space=func.input_domain, noise_prior=None, optimizer_restart=20
)
mfrbfgp.train(samples=sample_x, responses=sample_y)
# plot the land scape of  log_likelihood function
#  generate the meshgrid for the parameters theta and noise
theta = np.linspace(-4, 3, 100)
noise = np.linspace(0.01, 2, 100)
theta, noise = np.meshgrid(theta, noise)
# calculate the log_likelihood
log_likelihood = np.zeros(theta.shape)
for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        list_params = [theta[i, j], noise[i, j]]
        log_likelihood[i, j] = mfrbfgp._logLikelihood(params=list_params)

# plot the log_likelihood
fig, ax = plt.subplots()
cs = ax.contourf(theta, noise, log_likelihood, levels=50)
ax.set_xlabel("theta")
ax.set_ylabel("noise")
fig.colorbar(cs)
plt.savefig("mf_log_likelihood.png", dpi=300, bbox_inches="tight")
plt.show()
mfrbfgp_pred_y, mfrbfgp_pred_std = mfrbfgp.predict(
    x_predict=test_x, return_std=True)
# get prediction of low fidelity
pred_ly = mfrbfgp.predict_lf(test_xl=test_x)

print("mfrbfgp: ", mfrbfgp.kernel.param)
print("mfrbfgp: ", mfrbfgp.beta)
print("mfrbfgp: ", mfrbfgp.noise)
# plot
fig, ax = plt.subplots()
ax.plot(test_x, test_hy, label="high fidelity")
ax.plot(test_x, test_ly, label="low fidelity")
ax.plot(test_x, mfrbfgp_pred_y, label="mfrbfgp")
ax.fill_between(
    test_x.flatten(),
    (mfrbfgp_pred_y - 1.96 * mfrbfgp_pred_std).flatten(),
    (mfrbfgp_pred_y + 1.96 * mfrbfgp_pred_std).flatten(),
    alpha=0.5,
    label="95% confidence interval",
)
ax.plot(test_x, pred_ly, label="mfrbfgp lf")
ax.scatter(sample_x["hf"], sample_y["hf"], label="hf samples")
ax.scatter(sample_x["lf"], sample_y["lf"], label="lf samples")
ax.legend()
plt.show()
