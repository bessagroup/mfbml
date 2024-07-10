# create the configuration of the low-fidelity model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# get the accuracy metrics
from sklearn.metrics import r2_score

from mfbml.methods.bayes_neural_nets import BNNWrapper
from mfbml.metrics import (mean_log_likelihood_value, normalized_mae,
                           normalized_rmse)
from mfbml.problems.high_dimension_problems import MengCase1

# fix the random seed for reproducibility
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

# define function
func = MengCase1(noise_std=0.0)
# hf_samples = lf_samples[::10]  # sample every 5 points
hf_samples = torch.linspace(0.1, 0.9, 21).reshape(-1, 1)
# generate responses
hf_responses = func.hf(hf_samples, noise_hf=0.05)

# generate the test points
test_samples = torch.linspace(0, 1, 1001).reshape(-1, 1)
# noiseless responses
test_hf_responses_noiseless = func.hf(test_samples, noise_hf=0.0)
# noise responses
test_hf_responses = func.hf(test_samples, noise_hf=0.05)


# plot the function
fig, ax = plt.subplots()
ax.plot(test_samples, test_hf_responses_noiseless, label="HF")

# plot test noisy responses
# ax.scatter(test_samples, test_hf_responses, label="HF noise responses")
# ax.scatter(test_samples, test_lf_responses, label="LF noise responses")
# plot the samples
ax.scatter(hf_samples, hf_responses, label="HF samples")
plt.legend()
# save the figure
plt.savefig("sf_samples_plan.png", bbox_inches='tight', dpi=300)
plt.close()

# normalize the responses
hf_responses_mean = hf_responses.mean().numpy()
hf_responses_std = hf_responses.std().numpy()
hf_responses_scaled = (hf_responses - hf_responses_mean) / hf_responses_std

# create the sf_bnn model
sigma_scale = float(0.05 / hf_responses_std)
bnn_model = BNNWrapper(
    in_features=1,
    hidden_features=[50, 50],
    out_features=1,
    activation="Tanh",
    lr=0.001,
    sigma=sigma_scale,
)
# train the model
bnn_model.train(
    x=hf_samples,
    y=hf_responses_scaled,
    num_epochs=20000,
    sample_freq=100,
    burn_in_epochs=10000,
)
# save the model
torch.save(bnn_model, "bnn_model.pth")

# predict the test points
bnn_y, bnn_epistemic, bnn_total_unc, bnn_aleatoric = bnn_model.predict(
    x=test_samples)

# scale the predictions back
bnn_y = bnn_y * hf_responses_std + hf_responses_mean
bnn_total_unc = bnn_total_unc * hf_responses_std
bnn_aleatoric = bnn_aleatoric * hf_responses_std
bnn_epistemic = bnn_epistemic * hf_responses_std

# plot
plt.figure()
plt.plot(hf_samples, hf_responses, "o", label="hf")
plt.plot(test_samples.numpy(), bnn_y, label="hf prediction")
plt.plot(
    test_samples.numpy(),
    test_hf_responses_noiseless.numpy(),
    label="hf ground truth",
)
plt.fill_between(
    test_samples.flatten().numpy(),
    (bnn_y - 2 * bnn_total_unc).reshape(-1),
    (bnn_y + 2 * bnn_total_unc).reshape(-1),
    alpha=0.5,
    label="uncertainty",
)
plt.legend()
plt.savefig("sf_bnn.png", bbox_inches='tight', dpi=300)
plt.close()

# get the accuracy metrics
print("=====================================")
print("BNN")
nrmse_bnn = normalized_rmse(test_hf_responses_noiseless.numpy(),
                            bnn_y)
print("nrmse: ", nrmse_bnn)
nmae_bnn = normalized_mae(test_hf_responses_noiseless.numpy(),
                          bnn_y)
print("nmae: ", nmae_bnn)
ll_bnn = mean_log_likelihood_value(
    test_hf_responses.numpy(),
    bnn_y,
    bnn_total_unc)
print("ll_bnn: ", ll_bnn)
# r2 score
print("=====================================")
print("R2 Score")
r2_value = r2_score(test_hf_responses_noiseless.numpy(), bnn_y)
print("r2_value: ", r2_value)

# save the metrics
metrics = {
    "nrmse_bnn": nrmse_bnn,
    "nmae_bnn": nmae_bnn,
    "ll_bnn": ll_bnn,
    "r2_score": r2_value,
}
# save the metrics to csv

df = pd.DataFrame(metrics, index=[0])
df.to_csv("metrics.csv", index=False)
