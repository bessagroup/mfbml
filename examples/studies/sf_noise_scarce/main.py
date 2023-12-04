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
from f3dasm import (CategoricalParameter, ContinuousParameter,
                    DiscreteParameter, Domain, ExperimentData,
                    ExperimentSample)
from f3dasm.datageneration import DataGenerator
from mfpml.design_of_experiment.multifidelity_samplers import MFLatinHyperCube
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mfbml.get_methods.accuracy_metrics import log_likelihood_value
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
    problem_sets = ['Forrester_1a', 'mf_Bohachevsky', 'mf_Booth', 'mf_Borehole',
                    'mf_CurrinExp', 'mf_Hartman3', 'mf_Hartman6', 'mf_Himmelblau',
                    'mf_Park91A', 'mf_Park91B', 'mf_Sixhump']

    # define seed sets
    seed_sets = [i for i in range(1, 11)]

    # define the number of lf samples
    num_samples = [20 + 10*i for i in range(1, 10)]

    # define noise level for high fidelity
    noise_levels = [0.1, 0.3, 0.5]

    # create design variables
    design_variables = []
    for problem in problem_sets:
        for seed in seed_sets:
            for num_sample in num_samples:
                for noise_level in noise_levels:
                    design_variables.append(
                        [problem, seed, num_sample, noise_level])

    # save design variables tp pandas dataframe
    design_variables = pd.DataFrame(design_variables,
                                    columns=['problem', 'seed', 'num_sample', "noise_std"])

    # create the experiment data via f3dasm
    domain = Domain()
    domain.add('problem', CategoricalParameter(['Forrester_1a',
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
    domain.add('num_sample', DiscreteParameter(
        lower_bound=30, upper_bound=100))
    domain.add('noise_std', DiscreteParameter(
        lower_bound=1, upper_bound=2))

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=2970, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_sample'] = design_variables['num_sample']
    data.input_data.data['noise_std'] = design_variables['noise_std']

    # add output data
    data.add_output_parameter('progress')
    data.add_output_parameter('mae')
    data.add_output_parameter('mse')
    data.add_output_parameter('r2')
    data.add_output_parameter("log_likelihood")
    data.add_output_parameter("learned_noise_std")
    data.add_output_parameter('cpu_time')

    # save the experiment data
    data.store(filename='exp_{}'.format('gpr'))


def run_method(
        problem_name: str,
        seed: int,
        num_sample: int,
        noise_std: float) -> dict:
    """run a method on the data scarce noiseless problem sets, only fir multi-
    fidelity method. The single fidelity method is not supported.

    Parameters
    ----------
    problem_name : str
        name of the problem
    seed : int
        random seed
    num_sample : int
        number of samples
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
        num_hf_samples=num_sample*num_dim, num_lf_samples=num_sample*num_dim)
    sample_y = func.hf(sample_x['hf']) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=sample_x['hf'].shape[0]).reshape(-1, 1)

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
        model_name='gp', design_space=func.input_domain)

    start_time = time.time()
    model.train(sample_x=sample_x["hf"], sample_y=sample_y)
    pred_y, pred_std = model.predict(x_predict=test_x['hf'], return_std=True)
    end_time = time.time()
    cpu_time = end_time - start_time
    # accuracy test
    mae = mean_absolute_error(test_y, pred_y)
    mse = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    log_likelihood = log_likelihood_value(y_true=test_y_noise,
                                          y_pred_mean=pred_y,
                                          y_pred_std=pred_std)
    learned_noise_std = model.noise

    return {'mae': mae,
            'mse': mse,
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
        num_sample = sample.input_data['num_sample']
        noise_std = sample.input_data['noise_std']

        try:
            results = run_method(
                problem_name=problem,
                seed=seed,
                num_sample=num_sample,
                noise_std=noise_std
            )
            # print running information
            print("=====================================")
            print("problem: ", problem)
            print("seed: ", seed)
            print("num_sample: ", num_sample)
            print("noise_std: ", noise_std)
            # results print to the screen
            print("mae: ", results['mae'])
            print("mse: ", results['mse'])
            print("r2: ", results['r2'])
            print("cpu_time: ", results['cpu_time'])
            print("log_likelihood: ", results['log_likelihood'])
            print("learned_noise_std: ", results['learned_noise_std'])
            print("=====================================")
            # get the output data
            sample.output_data['progress'] = "finished"

            # update the output data
            sample.output_data['mae'] = results['mae']
            sample.output_data['mse'] = results['mse']
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
            sample.output_data['mae'] = None
            sample.output_data['mse'] = None
            sample.output_data['r2'] = None
            sample.output_data['cpu_time'] = None
            sample.output_data['log_likelihood'] = None
            sample.output_data['learned_noise_std'] = None

        return data

# function to execute the experiment data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('gpr'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='sequential')
    data.store(filename='exp_{}'.format('gpr_results'))


def main() -> None:
    """ Main script distinguishes between the master and the workers."""
    f3dasm.HPC_JOBID = 0
    if f3dasm.HPC_JOBID == 0:
        create_experiment_data()
        execute_experimentdata()
    elif f3dasm.HPC_JOBID > 0:
        sleep(3*f3dasm.HPC_JOBID)
        execute_experimentdata()


if __name__ == '__main__':
    main()
