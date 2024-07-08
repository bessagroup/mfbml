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
    # problem_sets = ['Forrester_1b']

    # define seed sets
    seed_sets = [1, 2, 3, 4, 5]

    # define the number of lf and hf samples
    num_lf_sample = [200]
    num_hf_sample = [5, 10, 15, 20, 25, 30]

    # # define noise level for high fidelity
    noise_levels = [0.3]

    # create design variables
    design_variables = []
    for problem in problem_sets:
        for seed in seed_sets:
            for num_lf in num_lf_sample:
                for num_hf in num_hf_sample:
                    for noise_std in noise_levels:
                        design_variables.append(
                            [ problem, seed, num_lf, num_hf, noise_std])

    # save design variables tp pandas data-frame
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
    data.sample(sampler='random', n_samples=300, seed=1)

    # replace the samples with the mesh_grid
    data.input_data.data['problem'] = design_variables['problem']
    data.input_data.data['seed'] = design_variables['seed']
    data.input_data.data['num_lf'] = design_variables['num_lf']
    data.input_data.data['num_hf'] = design_variables['num_hf']
    data.input_data.data['noise_std'] = design_variables['noise_std']

    # add output data for proposed method
    data.add_output_parameter('mkg_normalized_mae')
    data.add_output_parameter('mkg_normalized_rmse')
    data.add_output_parameter('mkg_r2')
    data.add_output_parameter("mkg_mean_log_likelihood")
    data.add_output_parameter("mkg_lf_training_time")
    data.add_output_parameter("mkg_hf_training_time")
    data.add_output_parameter('mkg_inference_time')
    data.add_output_parameter("mkg_learned_noise_std")
    #  add output data for cokriging
    data.add_output_parameter('ck_normalized_mae')
    data.add_output_parameter('ck_normalized_rmse')
    data.add_output_parameter('ck_r2')
    data.add_output_parameter("ck_mean_log_likelihood")
    data.add_output_parameter("ck_lf_training_time")
    data.add_output_parameter("ck_hf_training_time")
    data.add_output_parameter('ck_inference_time')
    data.add_output_parameter("ck_learned_noise_std")
    # add output data for hk
    data.add_output_parameter('hk_normalized_mae')
    data.add_output_parameter('hk_normalized_rmse')
    data.add_output_parameter('hk_r2')
    data.add_output_parameter("hk_mean_log_likelihood")
    data.add_output_parameter("hk_lf_training_time")
    data.add_output_parameter("hk_hf_training_time")
    data.add_output_parameter('hk_inference_time')
    data.add_output_parameter("hk_learned_noise_std")
    # add output data for scaled kriging
    data.add_output_parameter('sk_normalized_mae')
    data.add_output_parameter('sk_normalized_rmse')
    data.add_output_parameter('sk_r2')
    data.add_output_parameter("sk_mean_log_likelihood")
    data.add_output_parameter("sk_lf_training_time")
    data.add_output_parameter("sk_hf_training_time")
    data.add_output_parameter('sk_inference_time')
    data.add_output_parameter("sk_learned_noise_std")


    # save the experiment data
    data.store(filename='exp_{}'.format('noise_doe_experiments'))


def run_method(
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
        design_space=func._input_domain,
        num_fidelity=2,
        nested=False)
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
        design_space=func._input_domain,
        num_fidelity=2, nested=False)
    test_x = sampler.get_samples(
        num_samples=[1000 * num_dim, 1000 * num_dim])
    # get noiseless test samples
    test_y = func.hf(test_x[0])
    # get noise test samples
    np.random.seed(seed+1)
    test_y_noise = func.hf(test_x[0]) + \
        np.random.normal(loc=0, scale=noise_std,
                         size=1000*num_dim).reshape(-1, 1)

    # for proposed model =====================================================
    mkg_model = get_methods_for_noise_data(model_name="mf_rbf_gpr",
                                           design_space=func._input_domain)
    mkg_model.train(samples=sample_x,
                    responses=sample_y)
    start_time = time.time()
    mkg_pred_y, mkg_pred_std = mkg_model.predict(X=test_x[0], return_std=True)
    end_time = time.time()
    mkg_inference_time = end_time - start_time
    # accuracy test
    mkg_nmae = normalized_mae(test_y, mkg_pred_y)
    mkg_nrmse = normalized_rmse(test_y, mkg_pred_y)
    mkg_r2 = r2_score(test_y, mkg_pred_y)
    mkg_mll = mean_log_likelihood_value(y_true=test_y_noise,
                                                    y_pred_mean=mkg_pred_y,
                                                    y_pred_std=mkg_pred_std)
    # get noise estimation
    mkg_noise_std = mkg_model.noise
    # training time for low fidelity
    mkg_lf_training_time = mkg_model.lf_training_time
    # training time for high fidelity
    mkg_hf_training_time = mkg_model.hf_training_time

    # cokring model  ==========================================================
    ck_model = get_methods_for_noise_data(model_name="cokriging",
                                           design_space=func._input_domain)
    ck_model.train(samples=sample_x,
                   responses=sample_y)
    start_time = time.time()
    ck_pred_y, ck_pred_std = ck_model.predict(X=test_x[0], return_std=True)
    end_time = time.time()
    ck_inference_time = end_time - start_time
    # accuracy test
    ck_nmae = normalized_mae(test_y, ck_pred_y)
    ck_nrmse = normalized_rmse(test_y, ck_pred_y)
    ck_r2 = r2_score(test_y, ck_pred_y)
    ck_mll = mean_log_likelihood_value(y_true=test_y_noise,
                                        y_pred_mean=ck_pred_y,
                                        y_pred_std=ck_pred_std)
    # get noise estimation
    ck_noise_std = ck_model.noise
    # training time for low fidelity
    ck_lf_training_time = ck_model.lf_training_time
    # training time for high fidelity
    ck_hf_training_time = ck_model.hf_training_time

    # for hk model ============================================================
    hk_model = get_methods_for_noise_data(model_name="hk",
                                          design_space=func._input_domain)
    hk_model.train(samples=sample_x,
                   responses=sample_y)
    start_time = time.time()
    hk_pred_y, hk_pred_std = hk_model.predict(X=test_x[0], return_std=True)
    end_time = time.time()
    hk_inference_time = end_time - start_time
    # accuracy test
    hk_nmae = normalized_mae(test_y, hk_pred_y)
    hk_nrmse = normalized_rmse(test_y, hk_pred_y)
    hk_r2 = r2_score(test_y, hk_pred_y)
    hk_mll = mean_log_likelihood_value(y_true=test_y_noise,
                                        y_pred_mean=hk_pred_y,
                                        y_pred_std=hk_pred_std)
    # get noise estimation
    hk_noise_std = hk_model.noise
    # training time for low fidelity
    hk_lf_training_time = hk_model.lf_training_time
    # training time for high fidelity
    hk_hf_training_time = hk_model.hf_training_time


    # for scale kriging =======================================================
    sk_model = get_methods_for_noise_data(model_name="scaled_kriging",
                                          design_space=func._input_domain)
    sk_model.train(samples=sample_x,
                   responses=sample_y)
    start_time = time.time()
    sk_pred_y, sk_pred_std = sk_model.predict(X=test_x[0], return_std=True)
    end_time = time.time()
    sk_inference_time = end_time - start_time
    # accuracy test
    sk_nmae = normalized_mae(test_y, sk_pred_y)
    sk_nrmse = normalized_rmse(test_y, sk_pred_y)
    sk_r2 = r2_score(test_y, sk_pred_y)
    sk_mll = mean_log_likelihood_value(y_true=test_y_noise,
                                        y_pred_mean=sk_pred_y,
                                        y_pred_std=sk_pred_std)
    # get noise estimation
    sk_noise_std = sk_model.noise
    # training time for low fidelity
    sk_lf_training_time = sk_model.lf_training_time
    # training time for high fidelity
    sk_hf_training_time = sk_model.hf_training_time

    # record the results 
    results ={'mkg_normalized_mae': mkg_nmae,
            'mkg_normalized_rmse': mkg_nrmse,
            'mkg_r2': mkg_r2,
            'mkg_lf_training_time': mkg_lf_training_time,
            'mkg_hf_training_time': mkg_hf_training_time,
            'mkg_inference_time': mkg_inference_time,
            "mkg_mean_log_likelihood": mkg_mll,
            "mkg_learned_noise_std": mkg_noise_std,
            'ck_normalized_mae': ck_nmae,
            'ck_normalized_rmse': ck_nrmse,
            'ck_r2': ck_r2,
            'ck_lf_training_time': ck_lf_training_time,
            'ck_hf_training_time': ck_hf_training_time,
            'ck_inference_time': ck_inference_time,
            "ck_mean_log_likelihood": ck_mll,
            "ck_learned_noise_std": ck_noise_std,
            'hk_normalized_mae': hk_nmae,
            'hk_normalized_rmse': hk_nrmse,
            'hk_r2': hk_r2,
            'hk_lf_training_time': hk_lf_training_time,
            'hk_hf_training_time': hk_hf_training_time,
            'hk_inference_time': hk_inference_time,
            "hk_mean_log_likelihood": hk_mll,
            "hk_learned_noise_std": hk_noise_std,
            'sk_normalized_mae': sk_nmae,
            'sk_normalized_rmse': sk_nrmse,
            'sk_r2': sk_r2,
            'sk_lf_training_time': sk_lf_training_time,
            'sk_hf_training_time': sk_hf_training_time,
            'sk_inference_time': sk_inference_time,
            "sk_mean_log_likelihood": sk_mll,
            "sk_learned_noise_std": sk_noise_std}

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
        problem = sample.input_data['problem']
        seed = sample.input_data['seed']
        num_lf = sample.input_data['num_lf']
        num_hf = sample.input_data['num_hf']
        noise_std = sample.input_data['noise_std']
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
        print("mkg_normalized_mae: ", results['mkg_normalized_mae'])
        print("mkg_normalized_rmse: ", results['mkg_normalized_rmse'])
        print("mkg_r2: ", results['mkg_r2'])
        print("mkg_mean_log_likelihood: ", results['mkg_mean_log_likelihood'])
        print("mkg_learned_noise_std: ", results['mkg_learned_noise_std'])
        print("mkg_inference_time: ", results['mkg_inference_time'])
        print("mkg_lf_training_time: ", results['mkg_lf_training_time'])
        print("mkg_hf_training_time: ", results['mkg_hf_training_time'])

        print("ck_normalized_mae: ", results['ck_normalized_mae'])
        print("ck_normalized_rmse: ", results['ck_normalized_rmse'])
        print("ck_r2: ", results['ck_r2'])
        print("ck_mean_log_likelihood: ", results['ck_mean_log_likelihood'])
        print("ck_learned_noise_std: ", results['ck_learned_noise_std'])
        print("ck_inference_time: ", results['ck_inference_time'])
        print("ck_lf_training_time: ", results['ck_lf_training_time'])
        print("ck_hf_training_time: ", results['ck_hf_training_time'])

        print("hk_normalized_mae: ", results['hk_normalized_mae'])
        print("hk_normalized_rmse: ", results['hk_normalized_rmse'])
        print("hk_r2: ", results['hk_r2'])
        print("hk_mean_log_likelihood: ", results['hk_mean_log_likelihood'])
        print("hk_learned_noise_std: ", results['hk_learned_noise_std'])
        print("hk_inference_time: ", results['hk_inference_time'])
        print("hk_lf_training_time: ", results['hk_lf_training_time'])
        print("hk_hf_training_time: ", results['hk_hf_training_time'])

        print("sk_normalized_mae: ", results['sk_normalized_mae'])
        print("sk_normalized_rmse: ", results['sk_normalized_rmse'])
        print("sk_r2: ", results['sk_r2'])
        print("sk_mean_log_likelihood: ", results['sk_mean_log_likelihood'])
        print("sk_learned_noise_std: ", results['sk_learned_noise_std'])
        print("sk_inference_time: ", results['sk_inference_time'])
        print("sk_lf_training_time: ", results['sk_lf_training_time'])
        print("sk_hf_training_time: ", results['sk_hf_training_time'])


        


        print("=====================================")
        # update the output data
        sample.output_data['mkg_normalized_mae'] = results['mkg_normalized_mae']
        sample.output_data['mkg_normalized_rmse'] = results['mkg_normalized_rmse']
        sample.output_data['mkg_r2'] = results['mkg_r2']
        sample.output_data['mkg_mean_log_likelihood'] = results['mkg_mean_log_likelihood']
        sample.output_data['mkg_learned_noise_std'] = results['mkg_learned_noise_std']
        sample.output_data['mkg_inference_time'] = results['mkg_inference_time']
        sample.output_data['mkg_lf_training_time'] = results['mkg_lf_training_time']
        sample.output_data['mkg_hf_training_time'] = results['mkg_hf_training_time']

        sample.output_data['ck_normalized_mae'] = results['ck_normalized_mae']
        sample.output_data['ck_normalized_rmse'] = results['ck_normalized_rmse']
        sample.output_data['ck_r2'] = results['ck_r2']
        sample.output_data['ck_mean_log_likelihood'] = results['ck_mean_log_likelihood']
        sample.output_data['ck_learned_noise_std'] = results['ck_learned_noise_std']
        sample.output_data['ck_inference_time'] = results['ck_inference_time']
        sample.output_data['ck_lf_training_time'] = results['ck_lf_training_time']
        sample.output_data['ck_hf_training_time'] = results['ck_hf_training_time']

        sample.output_data['hk_normalized_mae'] = results['hk_normalized_mae']
        sample.output_data['hk_normalized_rmse'] = results['hk_normalized_rmse']
        sample.output_data['hk_r2'] = results['hk_r2']
        sample.output_data['hk_mean_log_likelihood'] = results['hk_mean_log_likelihood']
        sample.output_data['hk_learned_noise_std'] = results['hk_learned_noise_std']
        sample.output_data['hk_inference_time'] = results['hk_inference_time']
        sample.output_data['hk_lf_training_time'] = results['hk_lf_training_time']
        sample.output_data['hk_hf_training_time'] = results['hk_hf_training_time']

        sample.output_data['sk_normalized_mae'] = results['sk_normalized_mae']
        sample.output_data['sk_normalized_rmse'] = results['sk_normalized_rmse']
        sample.output_data['sk_r2'] = results['sk_r2']
        sample.output_data['sk_mean_log_likelihood'] = results['sk_mean_log_likelihood']
        sample.output_data['sk_learned_noise_std'] = results['sk_learned_noise_std']
        sample.output_data['sk_inference_time'] = results['sk_inference_time']
        sample.output_data['sk_lf_training_time'] = results['sk_lf_training_time']
        sample.output_data['sk_hf_training_time'] = results['sk_hf_training_time']

    
        return data


def execute_experimentdata() -> None:

    # load data from file
    data = f3dasm.ExperimentData.from_file(
        filename='exp_{}'.format('noise_doe_experiments'))
    # run the function
    data.evaluate(MFBMLExperiments(), mode='cluster')
    # data.store(filename='exp_{}'.format('noise_doe_experiments'))


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
