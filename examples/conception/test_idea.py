import matplotlib.pyplot as plt
import numpy as np
from mfpml.design_of_experiment.multifidelity_samplers import MFSobolSequence
from mfpml.models.kernels import RBF
from mfpml.problems.multifidelity_functions import Forrester_1b

from mfbml.mfrbfkriging import MFRBFKriging

# define function
func = Forrester_1b()

# define sampler
sampler = MFSobolSequence(design_space=func.design_space, seed=1)
sample_x = sampler.get_samples(num_hf_samples=4, num_lf_samples=11)
# update sample_x['hf']
sample_x['hf'] = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))

# get response
sample_y = func(sample_x)
# generate test samples
test_x = np.linspace(0, 1, 1000).reshape(-1, 1)
test_hy = func.hf(test_x)
test_ly = func.lf(test_x)

# define kernel
kernel = RBF(theta=np.zeros(1), bounds=[-1, 2])
model = MFRBFKriging(design_space=func.input_domain, kernel=kernel)
model.train(samples=sample_x, responses=sample_y)
pred_y, pred_std = model.predict(x_predict=test_x, return_std=True)
# get prediction of low fidelity
pred_ly = model.predict_lf(test_xl=test_x)

# plot
fig, ax = plt.subplots()
ax.plot(sample_x['hf'], sample_y['hf'], 'x', label='samples')
ax.plot(test_x, test_hy, 'r--', label='truth')
ax.plot(test_x, pred_y, 'b-', label='hf prediction')
ax.plot(test_x, test_ly, label='lf true')
ax.plot(test_x, pred_ly, label='lf prediction')
ax.plot(sample_x['lf'], sample_y['lf'], 'x', label='lf samples')
ax.fill_between(test_x.flatten(),
                (pred_y-2 * pred_std).flatten(),
                (pred_y+2*pred_std).flatten(),
                alpha=0.5, label='CI interval')
plt.legend()
plt.savefig('test_idea.png')
plt.show()
