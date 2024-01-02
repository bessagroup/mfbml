# this file contains the accuracy metrics for the gpr model and MF-RBF-GPR for
# evaluating the performance of the model. The accuracy metrics are the log
# marginal likelihood.
from typing import Any

import numpy as np


def log_likelihood_value(y_true: np.ndarray,
                         y_pred_mean: np.ndarray,
                         y_pred_std: np.ndarray) -> float | Any:
    """log likelihood value for Bayesian machine learning model

    Parameters
    ----------
    y_true : np.ndarray
        true values without noise
    y_pred_mean : np.ndarray
        predicted mean values from bayesian machine learning model
    y_pred_std : np.ndarray
        predicted standard deviation values from bayesian machine learning model

    Returns
    -------
    float | Any
        log likelihood value
    """
    obj = np.sum(-0.5 * np.log(2 * np.pi * y_pred_std**2) -
                 0.5 * (y_true - y_pred_mean)**2 / y_pred_std**2)

    return obj


def normalized_rmse(y_true: np.ndarray,
                    y_pred_mean: np.ndarray) -> np.ndarray:
    """normalized root mean squared error(NRMSE), which is used to evaluate the
    overall performance of the model.

    Parameters
    ----------
    y_true : np.ndarray
        true values without noise
    y_pred_mean : np.ndarray
        predicted mean values from bayesian machine learning model

    Returns
    -------
    np.ndarray
        normalized root mean squared error
    """

    return np.sqrt(np.mean((y_true - y_pred_mean)**2)) / np.mean(np.abs(y_true))


def normalized_mae(y_true: np.ndarray,
                   y_pred_mean: np.ndarray) -> np.ndarray:
    """normalized max absolute error(NMAE), which is used to evaluate the local
    performance of the model.

    Parameters
    ----------
    y_true : np.ndarray
        true values without noise
    y_pred_mean : np.ndarray
        predicted mean values from bayesian machine learning model

    Returns
    -------
    np.ndarray
        normalized max absolute error
    """

    return np.max(np.abs(y_true - y_pred_mean)) / np.mean(np.abs(y_true))
