
import pickle
import time
from time import sleep
from typing import Any, Dict, List, Tuple

import f3dasm
import numpy as np
import pandas as pd
import torch
from f3dasm import (CategoricalParameter, DiscreteParameter, Domain,
                    ExperimentData, ExperimentSample)
from f3dasm.datageneration import DataGenerator
from sklearn.metrics import r2_score

from mfbml.methods.bayes_neural_nets import BNNWrapper
from mfbml.methods.mf_dnn_bnn import SequentialMFBNN
from mfbml.methods.mf_dnn_lr_bnn import MFDNNBNN
from mfbml.metrics.accuracy_metrics import (mean_log_likelihood_value,
                                            normalized_mae, normalized_rmse)


def normalize_inputs(x: torch.Tensor,
                     design_space: torch.Tensor) -> torch.Tensor:
    """normalize the inputs

    Parameters
    ----------
    x : torch.Tensor
        original inputs
    design_space : torch.Tensor
        design space

    Returns
    -------
    torch.Tensor
        normalized inputs
    """
    x = (x - design_space[:, 0]) / \
        (design_space[:, 1] - design_space[:, 0])

    return x


def normalize_outputs(y: torch.Tensor) -> torch.Tensor:
    """normalize the outputs

    Parameters
    ----------
    y : torch.Tensor
        original outputs

    Returns
    -------
    torch.Tensor
        normalized outputs
    """
    y = (y - torch.mean(y)) / torch.std(y)
    return y


def bnn_single_run(seed: int) -> Dict:

    # fix the seed for torch and numpy
    np.random.seed(seed)
    torch.manual_seed(seed)
    # read data from ../data_generation/data.pkl
    data = pickle.load(open("../data_generation/data_100D_example.pkl", "rb"))

    # get the data
    hf_samples = data["hf_samples"]
    test_samples = data["test_samples"]
    hf_responses = data["hf_responses"]
    test_responses_noiseless = data["test_responses_noiseless"]
    test_responses_noisy = data["test_responses_noisy"]

    # design space
    design_space = torch.tile(torch.Tensor([-3, 3]), (hf_samples.shape[1], 1))

    # scale the data
    hf_samples_scaled = normalize_inputs(hf_samples, design_space)
    test_samples_scaled = normalize_inputs(test_samples, design_space)
    hf_responses_scaled = normalize_outputs(hf_responses)

    # scale the sigma
    sigma_scaled = float(10.0 / torch.std(hf_responses))
    # define the bnn model
    bnn_model = BNNWrapper(in_features=100,
                           hidden_features=[512, 512],
                           out_features=1,
                           activation="ReLU",
                           lr=0.001,
                           sigma=sigma_scaled)

    # train the bnn model
    bnn_model.train(x=hf_samples_scaled,
                    y=hf_responses_scaled,
                    num_epochs=50000,
                    sample_freq=100,
                    burn_in_epochs=10000,
                    print_info=True)

    # predict the MFDNNBNN object
    y, epistemic, total_unc, aleatoric = bnn_model.predict(
        x=test_samples_scaled)

    # scale the data back
    y = y * torch.std(hf_responses).numpy() + torch.mean(hf_responses).numpy()
    total_unc = total_unc * torch.std(hf_responses).numpy()
    aleatoric = aleatoric * torch.std(hf_responses).numpy()
    epistemic = epistemic * torch.std(hf_responses).numpy()

    # calculate the nmae, nrmse, r2 score and log likelihood
    nmae = normalized_mae(test_responses_noiseless.numpy(), y)
    nrmse = normalized_rmse(test_responses_noiseless.numpy(), y)
    r2 = r2_score(test_responses_noiseless.numpy(), y)

    # calculate the log likelihood
    log_likelihood = mean_log_likelihood_value(
        test_responses_noisy.numpy(), y, total_unc)

    # save the results
    results = {"nmae": nmae,
               "nrmse": nrmse,
               "r2": r2,
               "log_likelihood": log_likelihood,
               "lf_nmae": 0.0,
               "lf_nrmse": 0.0,
               "lf_r2": 0.0,
               "beta_0": 0.0,
               "beta_1": 0.0, }

    return results


def dnnlrbnn_single_run(seed: int) -> dict[str, Any]:

    # fix the seed for torch and numpy
    np.random.seed(seed)
    torch.manual_seed(seed)
    # read data from ../data_generation/data.pkl
    data = pickle.load(open("../data_generation/data_100D_example.pkl", "rb"))

    # get the data
    hf_samples = data["hf_samples"]
    lf_samples = data["lf_samples"]
    test_samples = data["test_samples"]
    hf_responses = data["hf_responses"]
    lf_responses = data["lf_responses"]
    test_responses_noiseless = data["test_responses_noiseless"]
    test_responses_noisy = data["test_responses_noisy"]
    lf_responses_noisy = data["lf_responses_noisy"]

    # design space
    design_space = torch.tile(torch.Tensor([-3, 3]), (hf_samples.shape[1], 1))
    # create the samples and responses dictionary
    samples = {"lf": lf_samples,
               "hf": hf_samples}
    responses = {"lf": lf_responses,
                 "hf": hf_responses}

    # create the configuration of the low-fidelity model
    lf_configure = {"in_features": 100,
                    "hidden_features": [256, 256],
                    "out_features": 1,
                    "activation": "Tanh",
                    "optimizer": "Adam",
                    "lr": 0.0001,
                    "weight_decay": 0.00003,
                    "loss": "mse"}

    # create the configuration of the high-fidelity model
    hf_configure = {"in_features": 100,
                    "hidden_features": [512, 512],
                    "out_features": 1,
                    "activation": "ReLU",
                    "lr": 0.001,
                    "sigma": 10}

    # create the MFDNNBNN object
    mfdnnbnn = MFDNNBNN(design_space=design_space,
                        lf_configure=lf_configure,
                        hf_configure=hf_configure,
                        beta_optimize=True, 
                        beta_bounds=[-100, 100],
                        discrepancy_normalization="diff")
    # lf train config
    lf_train_config = {"batch_size": 5000,
                       "num_epochs": 50000,
                       "print_iter": 100,
                       "data_split": True}
    hf_train_config = {"num_epochs": 50000,
                       "sample_freq": 100,
                       "print_info": True,
                       "burn_in_epochs": 10000}

    # train the MFDNNBNN object
    mfdnnbnn.train(samples=samples,
                   responses=responses,
                   lf_train_config=lf_train_config,
                   hf_train_config=hf_train_config
                   )

    # predict the MFDNNBNN object
    y, _, total_unc, _ = mfdnnbnn.predict(x=test_samples)
    # lf prediction
    lf_y = mfdnnbnn.predict_lf(x=test_samples, output_format="numpy")
    # calculate the nmae, nrmse, r2 score and log likelihood
    nmae = normalized_mae(test_responses_noiseless.numpy(), y)
    nrmse = normalized_rmse(test_responses_noiseless.numpy(), y)
    r2 = r2_score(test_responses_noiseless.numpy(), y)

    # calculate the log likelihood
    log_likelihood = mean_log_likelihood_value(
        test_responses_noisy.numpy(), y, total_unc)

    # lf nmae, nrmse, r2 score
    lf_nmae = normalized_mae(lf_responses_noisy.numpy(), lf_y)
    lf_nrmse = normalized_rmse(
        lf_responses_noisy.numpy(), lf_y)
    lf_r2 = r2_score(lf_responses_noisy.numpy(), lf_y)

    # save the results
    results = {"nmae": nmae,
               "nrmse": nrmse,
               "r2": r2,
               "log_likelihood": log_likelihood,
               "lf_nmae": lf_nmae,
               "lf_nrmse": lf_nrmse,
               "lf_r2": lf_r2,
               "beta_0": mfdnnbnn.beta[0],
               "beta_1": mfdnnbnn.beta[1], }

    return results


def dnn_bnn_single_run(seed: int) -> dict:

    # set the seed of torch and numpy
    np.random.seed(seed)
    torch.manual_seed(seed)
    # read data from ../data_generation/data.pkl
    data = pickle.load(open("../data_generation/data_100D_example.pkl", "rb"))
    print(data["hf_samples"].shape)
    # get the data
    hf_samples = data["hf_samples"]
    lf_samples = data["lf_samples"]
    test_samples = data["test_samples"]
    hf_responses = data["hf_responses"]
    lf_responses = data["lf_responses"]
    test_responses_noiseless = data["test_responses_noiseless"]
    test_responses_noisy = data["test_responses_noisy"]
    lf_responses_noisy = data["lf_responses_noisy"]

    # design space
    design_space = torch.tile(torch.Tensor([-3, 3]), (hf_samples.shape[1], 1))

    # create the samples and responses dictionary
    samples = {"lf": lf_samples,
               "hf": hf_samples}

    responses = {"lf": lf_responses,
                 "hf": hf_responses}

    # create the configuration of the low-fidelity model
    lf_configure = {"in_features": 100,
                    "hidden_features": [256, 256],
                    "out_features": 1,
                    "activation": "Tanh",
                    "optimizer": "Adam",
                    "lr": 0.0001,
                    "weight_decay": 0.00003,
                    "loss": "mse"}

    # create the configuration of the high-fidelity model
    hf_configure = {"in_features": 101,
                    "hidden_features": [512, 512],
                    "out_features": 1,
                    "activation": "ReLU",
                    "lr": 0.001,
                    "sigma": 10.0}

    # create the sequential mf bnn object
    mfbnn = SequentialMFBNN(design_space=design_space,
                            lf_configure=lf_configure,
                            hf_configure=hf_configure)

    # lf train config
    lf_train_config = {"batch_size": 10000,
                       "num_epochs": 50000,
                       "print_iter": 1000,
                       "data_split": True}
    hf_train_config = {"num_epochs": 50000,
                       "sample_freq": 100,
                       "print_info": True,
                       "burn_in_epochs": 10000}

    # train the MFDNNBNN object
    mfbnn.train(samples=samples,
                responses=responses,
                lf_train_config=lf_train_config,
                hf_train_config=hf_train_config
                )

    # predict the MFDNNBNN object
    y, _, total_unc, _ = mfbnn.predict(x=test_samples)
    # lf prediction
    lf_y = mfbnn.predict_lf(x=test_samples, output_format="numpy")

    # calculate the nmae, nrmse, r2 score and log likelihood
    nmae = normalized_mae(test_responses_noiseless.numpy(), y)
    nrmse = normalized_rmse(test_responses_noiseless.numpy(), y)
    r2 = r2_score(test_responses_noiseless.numpy(), y)

    # calculate the log likelihood
    log_likelihood = mean_log_likelihood_value(
        test_responses_noisy.numpy(), y, total_unc)

    # lf nmae, nrmse, r2 score
    lf_nmae = normalized_mae(lf_responses_noisy.numpy(), lf_y)
    lf_nrmse = normalized_rmse(
        lf_responses_noisy.numpy(), lf_y)
    lf_r2 = r2_score(lf_responses_noisy.numpy(), lf_y)

    # save the results
    results = {"nmae": nmae,
               "nrmse": nrmse,
               "r2": r2,
               "log_likelihood": log_likelihood,
               "lf_nmae": lf_nmae,
               "lf_nrmse": lf_nrmse,
               "lf_r2": lf_r2,
               "beta_0": 0,
               "beta_1": 0}

    return results


def create_experiment_data() -> None:
    """create experimental data, where the design variables are functions,
    methods, and seeds, and the response variables are mae, mse, r2, and cpu
    time. f3dasm is used here to create the experimental data, and later the
    experimental data will be used to run the methods.

    Notes
    -----
    1. the design variables are functions, methods, and seeds
    2. the response variables are mae, mse, r2, and cpu time
    """
    # define problem sets
    method_sets = ['dnn_lr_bnn',]

    # define seed sets
    seed_sets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # create design variables
    design_variables = []
    for method in method_sets:
        for seed in seed_sets:
            design_variables.append(
                [method, seed])

    # save design variables tp pandas data-frame
    design_variables = pd.DataFrame(design_variables,
                                    columns=['method', 'seed'])

    # create the experiment data via f3dasm
    domain = Domain()
    domain.add('method', CategoricalParameter(['dnn_bnn',
                                               'bnn',
                                               'dnn_lr_bnn',]))
    domain.add('seed', DiscreteParameter(lower_bound=1, upper_bound=10))

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=10, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['method'] = design_variables['method']
    data.input_data.data['seed'] = design_variables['seed']

    # add output data for proposed method
    data.add_output_parameter('nmae')
    data.add_output_parameter('nrmse')
    data.add_output_parameter('r2')
    data.add_output_parameter("log_likelihood")
    data.add_output_parameter("lf_nmae")
    data.add_output_parameter("lf_nrmse")
    data.add_output_parameter('lf_re')
    data.add_output_parameter("beta_0")
    data.add_output_parameter("beta_1")

    # save the experiment data
    data.store(filename='exp_{}'.format('100d_problem'))


def run_method(method: str,
               seed: int) -> Dict:

    if method == "dnn_bnn":
        results = dnn_bnn_single_run(seed=seed)
    elif method == "dnn_lr_bnn":
        results = dnnlrbnn_single_run(seed=seed)
    elif method == "bnn":
        results = bnn_single_run(seed=seed)
    else:
        raise KeyError("the method is not define")

    return results


class MFBMLExperiments(DataGenerator):
    """MFBMLExperiments is a class to run the experiments on the data scarce
    noiseless problem sets.

    Parameters
    ----------
    DataGenerator : DataGenerator
        data generator class from f3dasm
    """

    def execute(data: ExperimentSample) -> ExperimentSample:
        """execute the simulation for a single row of the DOE

        Parameters
        ----------
        data : ExperimentSample
            experiment sample from f3dasm

        Returns
        -------
        ExperimentSample
            experiment sample from f3dasm
        """

        # get one row of the DOE
        sample = data.experiment_sample
        # print job number to the screen
        print(f"job number {sample.job_number}")

        # get the design parameters of this row
        method = sample.input_data['method']
        seed = sample.input_data['seed']

        results = run_method(
            method=method,
            seed=seed,

        )
        # print running information
        print("=====================================")
        print("job_number:" + str(sample.job_number))
        print("method: ", method)
        print("seed: ", seed)

        print("=====================================")
        # update the output data
        sample.output_data['nmae'] = results['nmae']
        sample.output_data['nrmse'] = results['nrmse']
        sample.output_data['r2'] = results['r2']
        sample.output_data['log_likelihood'] = results['log_likelihood']
        sample.output_data['lf_nmae'] = results['lf_nmae']
        sample.output_data['lf_nrmse'] = results['lf_nrmse']
        sample.output_data['lf_re'] = results['lf_r2']
        sample.output_data['beta_0'] = results['beta_0']
        sample.output_data['beta_1'] = results['beta_1']

        return data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('100d_problem'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='cluster')


def main() -> None:
    """ Main script distinguishes between the master and the workers."""
    if f3dasm.HPC_JOBID is None:
        create_experiment_data()
        execute_experimentdata()
    elif f3dasm.HPC_JOBID == 0:
        create_experiment_data()
        execute_experimentdata()
    elif f3dasm.HPC_JOBID > 0:
        try:
            sleep(10*f3dasm.HPC_JOBID)
            execute_experimentdata()
        except:
            sleep(20*f3dasm.HPC_JOBID)
            execute_experimentdata()


if __name__ == '__main__':
    main()
