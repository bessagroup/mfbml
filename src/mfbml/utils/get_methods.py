# this script is used to generate the method sets for the data scarce noiseless
# problem sets
# ---
# Created Date: 2021-09-09 15:06:24

# third party library
from typing import Any

import numpy as np
from mfpml.models.co_kriging import CoKriging
from mfpml.models.gaussian_process import \
    GaussianProcessRegression as GaussianProcess
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.models.scale_kriging import ScaledKriging

# local library
from mfbml.methods.krr_lr_gpr import MFRBFGPR
from mfbml.methods.mf_rbf_kriging import MFRBFKriging


def get_method(method_name: str, design_space: np.ndarray) -> Any:
    """get and initialize a method for the data scarce noiseless problem sets

    Parameters
    ----------
    method_name : str
        name of selected method
    design_space : np.ndarray
        design space of the problem

    Returns
    -------
    Any
        a method object

    Raises
    ------
    ValueError
        method not found error
    """

    if method_name == 'kriging':
        return GaussianProcess(design_space=design_space, optimizer_restart=10, noise_prior=0.0)
    elif method_name == 'hk':
        return HierarchicalKriging(design_space=design_space, noise_prior=0.0,  optimizer_restart=10)
    elif method_name == 'ck':
        return CoKriging(design_space=design_space, noise_prior=0.0, optimizer_restart=10)
    elif method_name == 'mf_rbf':
        return MFRBFKriging(design_space=design_space, optimizer_restart=10)
    elif method_name == 'mf_scale':
        return ScaledKriging(design_space=design_space,  noise_prior=0.0, optimizer_restart=10)
    else:
        raise ValueError('method name not found')


def get_methods_for_noise_data(model_name: str,
                               design_space: np.ndarray) -> Any:
    """get and initialize a method for the data scarce noiseless problem sets

    Parameters
    ----------
    model_name : str
        name of selected model
    design_space : np.ndarray
        design space of the problem

    Returns
    -------
    Any
        a method object

    """

    if model_name == 'gp':
        return GaussianProcess(design_space=design_space,
                               noise_prior=None,
                               optimizer_restart=10)
    elif model_name == 'mf_rbf_gpr':
        return MFRBFGPR(design_space=design_space,
                        noise_prior=None,
                        optimizer_restart=10)
    elif model_name == 'cokriging':
        return CoKriging(design_space=design_space,
                         optimizer_restart=10,
                         noise_prior=None)
    elif model_name == 'hk':
        return HierarchicalKriging(design_space=design_space,
                                   optimizer_restart=10,
                                   noise_prior=None)
    elif model_name == 'scaled_kriging':
        return ScaledKriging(design_space=design_space,
                             optimizer_restart=10,
                             noise_prior=None)
    else:
        raise ValueError('model name not found')
