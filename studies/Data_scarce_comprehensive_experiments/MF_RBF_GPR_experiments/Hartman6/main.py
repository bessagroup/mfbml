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

from mfbml.get_methods.accuracy_metrics import (log_likelihood_value,
                                                normalized_mae,
                                                normalized_rmse)
from mfbml.get_methods.utils import get_methods_for_noise_data
from mfbml.problem_sets.noiseless_problems import register_problem

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
    problem_sets = ['mf_Hartman6']

    # define seed sets
    seed_sets = [i for i in range(1, 6)]

    # define the number of lf and hf samples
    num_lf_sample = [100 + 20*i for i in range(1, 11)]
    num_hf_sample = [5 + 5*i for i in range(0, 20)]

    # define noise level for high fidelity
    noise_levels = [0.1, 0.3, 0.5]

    # create design variables
    design_variables = []

    for problem in problem_sets:
        for seed in seed_sets:
            for num_lf in num_lf_sample:
                for num_hf in num_hf_sample:
                    for noise_std in noise_levels:
                        design_variables.append(
                            [problem, seed, num_lf, num_hf, noise_std])

    # save design variables tp pandas dataframe
    design_variables = pd.DataFrame(design_variables,
                                    columns=['problem', 'seed',
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
    domain.add('num_lf', DiscreteParameter(
        lower_bound=100, upper_bound=200))
    domain.add('num_hf', DiscreteParameter(
        lower_bound=10, upper_bound=200))
    domain.add('noise_std', DiscreteParameter(
        lower_bound=1, upper_bound=2))

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=3000, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_lf'] = design_variables['num_lf']
    data.input_data.data['num_hf'] = design_variables['num_hf']
    data.input_data.data['noise_std'] = design_variables['noise_std']

    # add output data
    data.add_output_parameter('progress')
    data.add_output_parameter('normalized_mae')
    data.add_output_parameter('normalized_rmse')
    data.add_output_parameter('r2')
    data.add_output_parameter("log_likelihood")
    data.add_output_parameter("learned_noise_std")
    data.add_output_parameter('cpu_time')

    # save the experiment data
    data.store(filename='exp_{}'.format('mf_rbf_gpr'))


def run_method(
        problem_name: str,
        seed: int,
        num_lf: int,
        num_hf: int,
        noise_std: float) -> dict:
    """run a method on the data scarce noiseless problem sets, only fir multi-
    fidelity method. The single fidelity method is not supported.

    Parameters
    ----------
    problem_name : str
        name of the problem
    seed : int
        random seed
    num_lf : int
        number of low fidelity samples
    num_hf : int
        number of high fidelity samples
    noise_std : float
        noise level


    Returns
    -------
    dict
        a dictionary of results
    """
    # define function
    func = register_problem(problem_name=problem_name)
    num_dim = func.num_dim

    # define sampler
    sampler = MFLatinHyperCube(design_space=func.design_space, seed=seed)
    sample_x = sampler.get_samples(
        num_hf_samples=num_hf*num_dim, num_lf_samples=num_lf*num_dim)
    # define noise samples
    sample_y = {}
    sample_y["hf"] = func.hf(sample_x['hf']) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=sample_x['hf'].shape[0]).reshape(-1, 1)
    sample_y["lf"] = func.lf(sample_x['lf']) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=sample_x['lf'].shape[0]).reshape(-1, 1)

    # generate test samples
    sampler = MFLatinHyperCube(design_space=func.design_space, seed=seed + 1)
    test_x = sampler.get_samples(
        num_hf_samples=1000 * num_dim, num_lf_samples=1000 * num_dim)
    # get noiseless test samples
    test_y = func.hf(test_x['hf'])
    # get noise test samples
    test_y_noise = func.hf(test_x['hf']) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=1000*num_dim).reshape(-1, 1)

    # define kernel
    model = get_methods_for_noise_data(
        model_name='mf_rbf_gp', design_space=func.input_domain)

    start_time = time.time()
    model.train(samples=sample_x, responses=sample_y)
    pred_y, pred_std = model.predict(x_predict=test_x['hf'], return_std=True)
    end_time = time.time()
    cpu_time = end_time - start_time
    # accuracy test
    nmae = normalized_mae(test_y, pred_y)
    nrmse = normalized_rmse(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    log_likelihood = log_likelihood_value(y_true=test_y_noise,
                                          y_pred_mean=pred_y,
                                          y_pred_std=pred_std)
    learned_noise_std = model.noise

    return {'normalized_mae': nmae,
            'normalized_rmse': nrmse,
            'r2': r2,
            'cpu_time': cpu_time,
            "log_likelihood": log_likelihood,
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
        problem = sample.input_data['problem']
        seed = sample.input_data['seed']
        num_lf = sample.input_data['num_lf']
        num_hf = sample.input_data['num_hf']
        noise_std = sample.input_data['noise_std']

        try:
            results = run_method(
                problem_name=problem,
                seed=seed,
                num_lf=num_lf,
                num_hf=num_hf,
                noise_std=noise_std
            )
            # print running information
            print("=====================================")
            print("problem: ", problem)
            print("seed: ", seed)
            print("num_lf: ", num_lf)
            print("num_hf: ", num_hf)
            print("noise_std: ", noise_std)
            # results print to the screen
            print("normalized mae: ", results['normalized_mae'])
            print("normalized rmse: ", results['normalized_rmse'])
            print("r2: ", results['r2'])
            print("cpu_time: ", results['cpu_time'])
            print("log_likelihood: ", results['log_likelihood'])
            print("learned_noise_std: ", results['learned_noise_std'])
            print("=====================================")
            # get the output data
            sample.output_data['progress'] = "finished"

            # update the output data
            sample.output_data['normalized_mae'] = results['normalized_mae']
            sample.output_data['normalized_rmse'] = results['normalized_rmse']
            sample.output_data['r2'] = results['r2']
            sample.output_data['cpu_time'] = results['cpu_time']
            sample.output_data['log_likelihood'] = results['log_likelihood']
            sample.output_data['learned_noise_std'] = results['learned_noise_std']

        except Exception as e:
            # if the job failed, then we set the progress to be failed
            sample.output_data['progress'] = "failed"
            # print the error message
            print(e)
            # save other output data to be None
            sample.output_data['normalized_mae'] = None
            sample.output_data['normalized_rmse'] = None
            sample.output_data['r2'] = None
            sample.output_data['cpu_time'] = None
            sample.output_data['log_likelihood'] = None
            sample.output_data['learned_noise_std'] = None

        return data

# function to execute the experiment data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('mf_rbf_gpr'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='cluster')
    # data.store(filename='exp_{}'.format('mf_rbf_gpr_results'))


def main() -> None:
    """ Main script distinguishes between the master and the workers."""
    # f3dasm.HPC_JOBID = 0
    if f3dasm.HPC_JOBID == 0:
        create_experiment_data()
        execute_experimentdata()
    elif f3dasm.HPC_JOBID > 0:
        try:
            sleep(3*f3dasm.HPC_JOBID)
            execute_experimentdata()
        except:
            sleep(3*f3dasm.HPC_JOBID)
            execute_experimentdata()          


if __name__ == '__main__':
    main()
