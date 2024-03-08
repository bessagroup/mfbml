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

from mfbml.methods.mf_rbf_kriging import MFRBFKriging
from mfbml.metrics.accuracy_metrics import normalized_mae, normalized_rmse
from mfbml.problems.low_dimension_problems import register_problem

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
                    'mf_Park91A',
                    'mf_Park91B',
                    'mf_Sixhump']

    # define seed sets
    seed_sets = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # define the number of lf and hf samples
    num_lf_sample = [100]
    num_hf_sample = [2 + 2*i for i in range(0, 10)]

    # create design variables
    design_variables = []

    for problem in problem_sets:
        for seed in seed_sets:
            for num_lf in num_lf_sample:
                for num_hf in num_hf_sample:
                    design_variables.append(
                        [problem, seed, num_lf, num_hf])

    # save design variables tp pandas dataframe
    design_variables = pd.DataFrame(design_variables,
                                    columns=['problem', 'seed',
                                             'num_lf', "num_hf"])

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

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=900, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_lf'] = design_variables['num_lf']
    data.input_data.data['num_hf'] = design_variables['num_hf']

    # add output data
    data.add_output_parameter('progress')
    data.add_output_parameter('linear_without_bias_normalized_mae')
    data.add_output_parameter('linear_without_bias_normalized_rmse')
    data.add_output_parameter('linear_without_bias_r2')
    data.add_output_parameter('linear_without_bias_cpu_time')
    data.add_output_parameter('linear_normalized_mae')
    data.add_output_parameter('linear_normalized_rmse')
    data.add_output_parameter('linear_r2')
    data.add_output_parameter('linear_cpu_time')
    data.add_output_parameter('quadratic_normalized_mae')
    data.add_output_parameter('quadratic_normalized_rmse')
    data.add_output_parameter('quadratic_r2')
    data.add_output_parameter('quadratic_cpu_time')
    # save the experiment data
    data.store(filename='exp_{}'.format('mf_rbf_gpr'))


def run_method(
        problem_name: str,
        seed: int,
        num_lf: int,
        num_hf: int,) -> dict:
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
    sample_y = func(sample_x)

    # generate test samples
    sampler = MFLatinHyperCube(design_space=func.design_space, seed=seed + 1)
    test_x = sampler.get_samples(
        num_hf_samples=1000 * num_dim, num_lf_samples=1000 * num_dim)
    # get noiseless test samples
    test_y = func.hf(test_x['hf'])

    # ==================== linear without bias model ==================== #
    linear_without_bias_model = MFRBFKriging(
        design_space=func.input_domain,
        lf_poly_order="linear_without_const",
        optimizer_restart=10)

    start_time = time.time()
    linear_without_bias_model.train(samples=sample_x, responses=sample_y)
    linear_without_bias_pred_y, linear_without_bias_std = linear_without_bias_model.predict(
        X=test_x['hf'], return_std=True)
    end_time = time.time()
    linear_without_bias_cpu_time = end_time - start_time
    # accuracy test
    linear_without_bias_nmae = normalized_mae(
        test_y, linear_without_bias_pred_y)
    linear_without_bias_nrmse = normalized_rmse(
        test_y, linear_without_bias_pred_y)
    linear_without_bias_r2 = r2_score(test_y, linear_without_bias_pred_y)

    # ==================== linear model ==================== #
    linear_model = MFRBFKriging(
        design_space=func.input_domain,
        lf_poly_order="linear",
        optimizer_restart=10)

    start_time = time.time()
    linear_model.train(sample_x, sample_y)
    linear_model_pred_y, linear_model_pred_std = linear_model.predict(
        test_x['hf'], return_std=True)
    end_time = time.time()
    sf_cpu_time = end_time - start_time
    # accuracy test
    linear_model_nmae = normalized_mae(test_y, linear_model_pred_y)
    linear_model_nrmse = normalized_rmse(test_y, linear_model_pred_y)
    linear_model_r2 = r2_score(test_y, linear_model_pred_y)
    # ==================== quadratic model ==================== #
    quadratic_model = MFRBFKriging(
        design_space=func.input_domain,
        lf_poly_order="quadratic",
        optimizer_restart=10)

    start_time = time.time()
    quadratic_model.train(sample_x, sample_y)
    quadratic_model_pred_y, quadratic_model_pred_std = quadratic_model.predict(
        test_x['hf'], return_std=True)
    end_time = time.time()
    quadratic_cpu_time = end_time - start_time
    # accuracy test
    quadratic_nmae = normalized_mae(test_y, quadratic_model_pred_y)
    quadratic_nrmse = normalized_rmse(test_y, quadratic_model_pred_y)
    quadratic_r2 = r2_score(test_y, quadratic_model_pred_y)

    return {
        'linear_without_bias_normalized_mae': linear_without_bias_nmae,
        'linear_without_bias_normalized_rmse': linear_without_bias_nrmse,
        'linear_without_bias_r2': linear_without_bias_r2,
        'linear_without_bias_cpu_time': linear_without_bias_cpu_time,
        'linear_normalized_mae': linear_model_nmae,
        'linear_normalized_rmse': linear_model_nrmse,
        'linear_r2': linear_model_r2,
        'linear_cpu_time': sf_cpu_time,
        'quadratic_normalized_mae': quadratic_nmae,
        'quadratic_normalized_rmse': quadratic_nrmse,
        'quadratic_r2': quadratic_r2,
        'quadratic_cpu_time': quadratic_cpu_time,
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

        try:
            results = run_method(
                problem_name=problem,
                seed=seed,
                num_lf=num_lf,
                num_hf=num_hf,
            )
            # print running information
            print("=====================================")
            print("problem: ", problem)
            print("seed: ", seed)
            print("num_lf: ", num_lf)
            print("num_hf: ", num_hf)
            # results print to the screen
            print("linear_without_bias_normalized_mae: ",
                  results['linear_without_bias_normalized_mae'])
            print("linear_without_bias_normalized_rmse: ",
                  results['linear_without_bias_normalized_rmse'])
            print("linear_without_bias_r2: ",
                  results['linear_without_bias_r2'])
            print("linear_without_bias_cpu_time: ",
                  results['linear_without_bias_cpu_time'])
            print("linear_normalized_mae: ", results['linear_normalized_mae'])
            print("linear_normalized_rmse: ",
                  results['linear_normalized_rmse'])
            print("linear_r2: ", results['linear_r2'])
            print("linear_cpu_time: ", results['linear_cpu_time'])
            print("quadratic_normalized_mae: ",
                  results['quadratic_normalized_mae'])
            print("quadratic_normalized_rmse: ",
                  results['quadratic_normalized_rmse'])
            print("quadratic_r2: ", results['quadratic_r2'])
            print("quadratic_cpu_time: ", results['quadratic_cpu_time'])

            print("=====================================")
            # get the output data
            sample.output_data['progress'] = "finished"

            # update the output data
            sample.output_data['linear_without_bias_normalized_mae'] = results['linear_without_bias_normalized_mae']
            sample.output_data['linear_without_bias_normalized_rmse'] = results['linear_without_bias_normalized_rmse']
            sample.output_data['linear_without_bias_r2'] = results['linear_without_bias_r2']
            sample.output_data['linear_without_bias_cpu_time'] = results['linear_without_bias_cpu_time']
            sample.output_data['linear_normalized_mae'] = results['linear_normalized_mae']
            sample.output_data['linear_normalized_rmse'] = results['linear_normalized_rmse']
            sample.output_data['linear_r2'] = results['linear_r2']
            sample.output_data['linear_cpu_time'] = results['linear_cpu_time']
            sample.output_data['quadratic_normalized_mae'] = results['quadratic_normalized_mae']
            sample.output_data['quadratic_normalized_rmse'] = results['quadratic_normalized_rmse']
            sample.output_data['quadratic_r2'] = results['quadratic_r2']
            sample.output_data['quadratic_cpu_time'] = results['quadratic_cpu_time']

        except Exception as e:
            # if the job failed, then we set the progress to be failed
            sample.output_data['progress'] = "failed"
            # print the error message
            print(e)
            # save other output data to be None
            sample.output_data['linear_without_bias_normalized_mae'] = None
            sample.output_data['linear_without_bias_normalized_rmse'] = None
            sample.output_data['linear_without_bias_r2'] = None
            sample.output_data['linear_without_bias_cpu_time'] = None
            sample.output_data['linear_normalized_mae'] = None
            sample.output_data['linear_normalized_rmse'] = None
            sample.output_data['linear_r2'] = None
            sample.output_data['linear_cpu_time'] = None
            sample.output_data['quadratic_normalized_mae'] = None
            sample.output_data['quadratic_normalized_rmse'] = None
            sample.output_data['quadratic_r2'] = None
            sample.output_data['quadratic_cpu_time'] = None
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
    f3dasm.HPC_JOBID = 0
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
