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
from mfpml.design_of_experiment.mf_samplers import MFLatinHyperCube
from sklearn.metrics import r2_score

from mfbml.metrics.accuracy_metrics import normalized_mae, normalized_rmse
from mfbml.problems.low_dimension_problems import register_problem
from mfbml.utils.get_methods import get_method

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
    problem_sets = [
        'mf_Bohachevsky', 'mf_Booth', 'mf_Borehole',
        'mf_CurrinExp', 'mf_Hartman3', 'mf_Hartman6',
        'mf_Himmelblau', 'mf_Park91A', 'mf_Park91B',
        'mf_Sixhump']

    # define method sets
    method_sets = ['mf_scale', 'mf_rbf', 'hk', 'ck']

    # define seed sets
    seed_sets = [i for i in range(1, 6)]

    # define the number of lf samples
    num_lf_sample = [20 + 5*i for i in range(1, 21)]
    num_hf_sample = [2*i for i in range(1, 11)]

    # create design variables
    design_variables = []
    for problem in problem_sets:
        for method in method_sets:
            for seed in seed_sets:
                for num_lf in num_lf_sample:
                    for num_hf in num_hf_sample:
                        design_variables.append(
                            [problem, method, seed, num_lf, num_hf])

    # save design variables tp pandas dataframe
    design_variables = pd.DataFrame(design_variables,
                                    columns=['problem', 'method', 'seed', 'num_lf', 'num_hf'])

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
    domain.add('method', CategoricalParameter(
        ['hk', 'ck', 'mf_rbf', 'mf_scale']))
    domain.add('seed', DiscreteParameter(lower_bound=1, upper_bound=10))
    domain.add('num_lf', DiscreteParameter(
        lower_bound=10, upper_bound=200))
    domain.add('num_hf', DiscreteParameter(
        lower_bound=2, upper_bound=20))

    # create the experiment data
    data = ExperimentData(domain=domain)
    data.sample(sampler='random', n_samples=40000, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['method'] = design_variables['method']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_lf'] = design_variables['num_lf']
    data.input_data.data['num_hf'] = design_variables['num_hf']

    # add output data
    data.add_output_parameter('progress')
    data.add_output_parameter('normalized_mae')
    data.add_output_parameter('normalized_rmse')
    data.add_output_parameter('r2')
    data.add_output_parameter('cpu_time')

    # save the experiment data
    data.store(filename='exp_{}'.format('noiseless_experiments'))


def run_method(method_name: str,
               problem_name: str,
               seed: int,
               num_hf_sample: int,
               num_lf_sample: int) -> dict:
    """run a method on the data scarce noiseless problem sets, only fir multi-
    fidelity method. The single fidelity method is not supported.

    Parameters
    ----------
    method_name : str
        name of selected method
    design_space : np.ndarray
        design space of the problem
    problem_name : str
        name of the problem
    seed : int
        random seed
    num_hf_sample : int
        number of high fidelity samples, the number is associated with dimension
    num_lf_sample : int
        number of low fidelity samples, the number is associated with dimension

    Returns
    -------
    dict
        a dictionary of results
    """
    # define function
    func = register_problem(problem_name=problem_name)
    num_dim = func.num_dim

    # define sampler
    sampler = MFLatinHyperCube(design_space=func.input_domain, num_fidelity=2)
    sample_x = sampler.get_samples(
        num_samples=[num_hf_sample, num_lf_sample], seed=seed)
    sample_y = func(sample_x)

    # generate test samples
    test_x = sampler.get_samples(
        num_samples=[100*num_dim, 100*num_dim], seed=seed + 1)

    # define kernel
    model = get_method(method_name=method_name, design_space=func.input_domain)

    start_time = time.time()
    model.train(sample_x, sample_y)
    pred_y = model.predict(test_x[0], return_std=False)
    end_time = time.time()
    # print('prediction time: ', end_time - start_time)
    cpu_time = end_time - start_time
    # accuracy test
    nmae = normalized_mae(func.hf(test_x[0]), pred_y)
    nrmse = normalized_rmse(func.hf(test_x[0]), pred_y)
    r2 = r2_score(func.hf(test_x[0]), pred_y)

    return {'normalized_mae': nmae,
            'normalized_rmse': nrmse,
            'r2': r2,
            'cpu_time': cpu_time}


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
        method = sample.input_data['method']
        seed = sample.input_data['seed']
        num_lf = sample.input_data['num_lf']
        num_hf = sample.input_data['num_hf']

        try:
            results = run_method(method_name=method,
                                 problem_name=problem,
                                 seed=seed,
                                 num_hf_sample=num_hf,
                                 num_lf_sample=num_lf)
            # print running information
            print("=====================================")
            print("problem: ", problem)
            print("method: ", method)
            print("seed: ", seed)
            print("num_lf: ", num_lf)
            print("num_hf: ", num_hf)
            # results print to the screen
            print("normalized mae: ", results['normalized_mae'])
            print("normalized rmse: ", results['normalized_rmse'])
            print("r2: ", results['r2'])
            print("cpu_time: ", results['cpu_time'])
            print("=====================================")
            # get the output data
            sample.output_data['progress'] = "finished"

            # update the output data
            sample.output_data['normalized_mae'] = results['normalized_mae']
            sample.output_data['normalized_rmse'] = results['normalized_rmse']
            sample.output_data['r2'] = results['r2']
            sample.output_data['cpu_time'] = results['cpu_time']

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

        return data

# function to execute the experiment data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('noiseless_experiments'))
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
