import numpy as np


def forrestor_hf(x):
    """
    The Forrestor function
    """
    return - (6 * x - 2)**2 * np.sin(12 * x - 4)


def forrestor_1a(x):