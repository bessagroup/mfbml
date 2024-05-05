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
                    'mf_Booth',
                    'mf_Borehole',
                    'mf_CurrinExp',
                    'mf_Hartman3',
                    'mf_Hartman6',
                    'mf_Himmelblau',
                    'mf_Park91A',
                    'mf_Park91B',
                    'mf_Sixhump']

    # define seed sets
    seed_sets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # define the number of lf and hf samples
    num_lf_sample = [200]
    num_hf_sample = [5 + 5*i for i in range(0, 10)]

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
    domain.add('problem', CategoricalParameter([
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
    data.add_output_parameter('mf_normalized_mae')
    data.add_output_parameter('mf_normalized_rmse')
    data.add_output_parameter('mf_r2')
    data.add_output_parameter("mf_mean_log_likelihood")
    data.add_output_parameter("mf_learned_noise_std")
    data.add_output_parameter('mf_cpu_time')
    data.add_output_parameter('sf_normalized_mae')
    data.add_output_parameter('sf_normalized_rmse')
    data.add_output_parameter('sf_r2')
    data.add_output_parameter("sf_mean_log_likelihood")
    data.add_output_parameter("sf_learned_noise_std")
    data.add_output_parameter('sf_cpu_time')

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

    # train multi-fidelity model
    mf_model = get_methods_for_noise_data(
        model_name='mf_rbf_gp', design_space=func.input_domain)

    start_time = time.time()
    mf_model.train(samples=sample_x, responses=sample_y)
    mf_pred_y, mf_pred_std = mf_model.predict(
        X=test_x['hf'], return_std=True)
    end_time = time.time()
    mf_cpu_time = end_time - start_time
    # accuracy test
    mf_nmae = normalized_mae(test_y, mf_pred_y)
    mf_nrmse = normalized_rmse(test_y, mf_pred_y)
    mf_r2 = r2_score(test_y, mf_pred_y)
    mf_mean_log_likelihood = mean_log_likelihood_value(y_true=test_y_noise,
                                                       y_pred_mean=mf_pred_y,
                                                       y_pred_std=mf_pred_std)
    mf_learned_noise_std = mf_model.noise

    # single fidelity model
    sf_model = get_methods_for_noise_data(
        model_name='gp', design_space=func.input_domain)

    start_time = time.time()
    sf_model.train(sample_x['hf'], sample_y['hf'])
    sf_pred_y, sf_pred_std = sf_model.predict(test_x['hf'], return_std=True)
    end_time = time.time()
    sf_cpu_time = end_time - start_time
    # accuracy test
    sf_nmae = normalized_mae(test_y, sf_pred_y)
    sf_nrmse = normalized_rmse(test_y, sf_pred_y)
    sf_r2 = r2_score(test_y, sf_pred_y)
    sf_mean_log_likelihood = mean_log_likelihood_value(y_true=test_y_noise,
                                                       y_pred_mean=sf_pred_y,
                                                       y_pred_std=sf_pred_std)
    sf_learned_noise_std = sf_model.noise

    return {
        "mf_normalized_mae": mf_nmae,
        "mf_normalized_rmse": mf_nrmse,
        "mf_r2": mf_r2,
        "mf_cpu_time": mf_cpu_time,
        "mf_mean_log_likelihood": mf_mean_log_likelihood,
        "mf_learned_noise_std": mf_learned_noise_std,
        "sf_normalized_mae": sf_nmae,
        "sf_normalized_rmse": sf_nrmse,
        "sf_r2": sf_r2,
        "sf_cpu_time": sf_cpu_time,
        "sf_mean_log_likelihood": sf_mean_log_likelihood,
        "sf_learned_noise_std": sf_learned_noise_std
    }


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
            print("mf_normalized_mae: ", results['mf_normalized_mae'])
            print("mf_normalized_rmse: ", results['mf_normalized_rmse'])
            print("mf_r2: ", results['mf_r2'])
            print("mf_cpu_time: ", results['mf_cpu_time'])
            print("mf learned noise:", results['mf_learned_noise_std'])
            print("mf_mean_log_likelihood: ",
                  results['mf_mean_log_likelihood'])
            print("sf_normalized_mae: ", results['sf_normalized_mae'])
            print("sf_normalized_rmse: ", results['sf_normalized_rmse'])
            print("sf_r2: ", results['sf_r2'])
            print("sf_cpu_time: ", results['sf_cpu_time'])
            print("sf learned noise:", results['sf_learned_noise_std'])
            print("sf_mean_log_likelihood: ",
                  results['sf_mean_log_likelihood'])

            print("=====================================")
            # get the output data
            sample.output_data['progress'] = "finished"

            # update the output data
            sample.output_data['mf_normalized_mae'] = results['mf_normalized_mae']
            sample.output_data['mf_normalized_rmse'] = results['mf_normalized_rmse']
            sample.output_data['mf_r2'] = results['mf_r2']
            sample.output_data['mf_cpu_time'] = results['mf_cpu_time']
            sample.output_data['mf_mean_log_likelihood'] = results['mf_mean_log_likelihood']
            sample.output_data['mf_learned_noise_std'] = results['mf_learned_noise_std']
            sample.output_data['sf_normalized_mae'] = results['sf_normalized_mae']
            sample.output_data['sf_normalized_rmse'] = results['sf_normalized_rmse']
            sample.output_data['sf_r2'] = results['sf_r2']
            sample.output_data['sf_cpu_time'] = results['sf_cpu_time']
            sample.output_data['sf_mean_log_likelihood'] = results['sf_mean_log_likelihood']
            sample.output_data['sf_learned_noise_std'] = results['sf_learned_noise_std']

        except Exception as e:
            # if the job failed, then we set the progress to be failed
            sample.output_data['progress'] = "failed"
            # print the error message
            print(e)
            # save other output data to be None
            sample.output_data['mf_normalized_mae'] = None
            sample.output_data['mf_normalized_rmse'] = None
            sample.output_data['mf_r2'] = None
            sample.output_data['mf_cpu_time'] = None
            sample.output_data['mf_mean_log_likelihood'] = None
            sample.output_data['mf_learned_noise_std'] = None
            sample.output_data['sf_normalized_mae'] = None
            sample.output_data['sf_normalized_rmse'] = None
            sample.output_data['sf_r2'] = None
            sample.output_data['sf_cpu_time'] = None
            sample.output_data['sf_mean_log_likelihood'] = None
            sample.output_data['sf_learned_noise_std'] = None

        return data

# function to execute the experiment data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('mf_rbf_gpr'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='cluster')
    # save the data
    # data.store(filename='exp_{}'.format('mf_rbf_gpr_results'))


def main() -> None:
    """ Main script distinguishes between the master and the workers."""
    # f3dasm.HPC_JOBID = 0
    if f3dasm.HPC_JOBID == 0:
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
