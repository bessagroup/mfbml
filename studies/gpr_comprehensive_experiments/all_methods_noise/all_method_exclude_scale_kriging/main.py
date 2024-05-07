# this script used to run different methods on the data scarce noiseless problem
# sets

#                                                                       Modules
# =============================================================================
# standard library

import time
from time import sleep
from typing import Any

import f3dasm
import numpy as np
import pandas as pd
from f3dasm import (CategoricalParameter, DiscreteParameter, Domain,
                    ExperimentData, ExperimentSample)
from f3dasm.datageneration import DataGenerator
from mfpml.design_of_experiment.multifidelity_samplers import MFLatinHyperCube
from sklearn.metrics import r2_score

from mfbml.metrics.accuracy_metrics import (mean_log_likelihood_value,
                                            normalized_mae, normalized_rmse)
from mfbml.problems.low_dimension_problems import register_problem
from mfbml.utils.get_methods import get_methods_for_noise_data

# ===========================================================================


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
    problem_sets = ['mf_Bohachevsky',
                    'Forrester_1b',
                    'mf_Booth',
                    'mf_Borehole',
                    'mf_CurrinExp',
                    'mf_Hartman3',
                    'mf_Hartman6',
                    'mf_Park91A',
                    'mf_Park91B',
                    'mf_Sixhump']

    # define seed sets
    seed_sets = [1 , 2, 3, 4, 5]

    # define the number of lf and hf samples
    num_lf_sample = [50, 75, 100, 125, 150, 175,200]
    num_hf_sample = [5, 10, 15, 20, 25, 30]

    # # define noise level for high fidelity
    noise_levels = [0.1, 0.3, 0.5]


    # method sets
    method_sets = ['cokriging','mf_rbf_gpr', 'scaled_kriging','hk']

    # create design variables
    design_variables = []
    for method in method_sets:
        for problem in problem_sets:
            for seed in seed_sets:
                for num_lf in num_lf_sample:
                    for num_hf in num_hf_sample:
                        for noise_std in noise_levels:
                            design_variables.append(
                                [method, problem, seed, num_lf, num_hf, noise_std])

    # save design variables tp pandas data-frame
    design_variables = pd.DataFrame(design_variables,
                                    columns=['method', 'problem', 'seed',
                                             'num_lf', "num_hf", "noise_std"])

    # create the experiment data via f3dasm
    domain = Domain()
    domain.add('problem', CategoricalParameter(['Forrester_1a',
                                                'Forrester_1b',
                                                'Forrester_1c',
                                                'mf_Bohachevsky',
                                                'mf_Booth',
                                                'mf_Borehole',
                                                'mf_CurrinExp',
                                                'mf_Hartman3',
                                                'mf_Hartman6',
                                                'mf_Himmelblau',
                                                'mf_Park91A',
                                                'mf_Park91B',
                                                'mf_Sixhump']))
    domain.add('seed', DiscreteParameter(lower_bound=1, upper_bound=10))
    domain.add('method', CategoricalParameter(['mf_rbf_gpr',
                                               'scaled_kriging',
                                               'hk',
                                               'cokriging']))
    domain.add('num_lf', DiscreteParameter(
        lower_bound=100, upper_bound=200))
    domain.add('num_hf', DiscreteParameter(
        lower_bound=10, upper_bound=200))
    domain.add('noise_std', DiscreteParameter(
        lower_bound=1, upper_bound=2))

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=25200, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['method'] = design_variables['method']
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_lf'] = design_variables['num_lf']
    data.input_data.data['num_hf'] = design_variables['num_hf']
    data.input_data.data['noise_std'] = design_variables['noise_std']

    # add output data
    data.add_output_parameter('normalized_mae')
    data.add_output_parameter('normalized_rmse')
    data.add_output_parameter('r2')
    data.add_output_parameter("mean_log_likelihood")
    data.add_output_parameter("lf_training_time")
    data.add_output_parameter("hf_training_time")
    data.add_output_parameter('inference_time')
    data.add_output_parameter("learned_noise_std")

    # save the experiment data
    data.store(filename='exp_{}'.format('noise_doe_experiments'))


def run_method(
        method_name: str,
        problem_name: str,
        seed: int,
        num_lf: int,
        num_hf: int,
        noise_std: float) -> dict:
    # define function
    func = register_problem(problem_name=problem_name)
    num_dim = func.num_dim

    # define sampler
    sampler = MFLatinHyperCube(
        design_space=func._input_domain, num_fidelity=2, nested=False)
    sample_x = sampler.get_samples(num_samples=[num_hf*num_dim, num_lf*num_dim],
                                   seed=seed)
    # define noise samples
    # fix the seed for noise samples
    np.random.seed(seed)
    # get noise samples
    sample_y = {}
    sample_y[0] = func.hf(sample_x[0]) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=sample_x[0].shape[0]).reshape(-1, 1)
    sample_y[1] = func.lf(sample_x[1]) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=sample_x[1].shape[0]).reshape(-1, 1)

    # generate test samples
    sampler = MFLatinHyperCube(
        design_space=func._input_domain, num_fidelity=2, nested=False)
    test_x = sampler.get_samples(
        num_samples=[1000 * num_dim, 1000 * num_dim])
    # get noiseless test samples
    test_y = func.hf(test_x[0])
    # get noise test samples
    np.random.seed(seed+1)
    test_y_noise = func.hf(test_x[0]) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=1000*num_dim).reshape(-1, 1)

    # define kernel
    model = get_methods_for_noise_data(
        model_name=method_name, design_space=func._input_domain)

    model.train(samples=sample_x, responses=sample_y)
    start_time = time.time()
    pred_y, pred_std = model.predict(X=test_x[0], return_std=True)
    end_time = time.time()
    inference_time = end_time - start_time
    # accuracy test
    nmae = normalized_mae(test_y, pred_y)
    nrmse = normalized_rmse(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    mean_log_likelihood = mean_log_likelihood_value(y_true=test_y_noise,
                                               y_pred_mean=pred_y,
                                               y_pred_std=pred_std)
    # get noise estimation
    learned_noise_std = model.noise
    # training time for low fidelity
    lf_training_time = model.lf_training_time
    # training time for high fidelity
    hf_training_time = model.hf_training_time

    return {'normalized_mae': nmae,
            'normalized_rmse': nrmse,
            'r2': r2,
            'lf_training_time': lf_training_time,
            'hf_training_time': hf_training_time,
            'inference_time': inference_time,
            "mean_log_likelihood": mean_log_likelihood,
            "learned_noise_std": learned_noise_std}


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
        method_name = sample.input_data['method']
        problem = sample.input_data['problem']
        seed = sample.input_data['seed']
        num_lf = sample.input_data['num_lf']
        num_hf = sample.input_data['num_hf']
        noise_std = sample.input_data['noise_std']
        results = run_method(
            method_name=method_name,
            problem_name=problem,
            seed=seed,
            num_lf=num_lf,
            num_hf=num_hf,
            noise_std=noise_std
        )
        # print running information
        print("=====================================")
        print("method: ", method_name)
        print("problem: ", problem)
        print("seed: ", seed)
        print("num_lf: ", num_lf)
        print("num_hf: ", num_hf)
        print("noise_std: ", noise_std)
        # results print to the screen
        print("normalized mae: ", results['normalized_mae'])
        print("normalized rmse: ", results['normalized_rmse'])
        print("r2: ", results['r2'])
        print("lf_training_time: ", results['lf_training_time'])
        print("hf_training_time: ", results['hf_training_time'])
        print("inference_time: ", results['inference_time'])
        print("mean_log_likelihood: ", results['mean_log_likelihood'])
        print("learned_noise_std: ", results['learned_noise_std'])
        print("=====================================")
        # update the output data
        sample.output_data['normalized_mae'] = results['normalized_mae']
        sample.output_data['normalized_rmse'] = results['normalized_rmse']
        sample.output_data['r2'] = results['r2']
        sample.output_data['inference_time'] = results['inference_time']
        sample.output_data['mean_log_likelihood'] = results['mean_log_likelihood']
        sample.output_data['learned_noise_std'] = results['learned_noise_std']
        sample.output_data['lf_training_time'] = results['lf_training_time']
        sample.output_data['hf_training_time'] = results['hf_training_time']

        return data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('noise_doe_experiments'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='cluster')
    # data.store(filename='exp_{}'.format('mf_rbf_gpr_results'))


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
