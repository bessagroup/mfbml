
# ------------------ Beginning of Reference Python Module ---------------------
"""This script is used prepare problem for low-dimensional problems, the
multi-fidelity problem problems are originally from the repository:
https: // github.com/JiaxiangYi96/mfpml.git

"""

#                                                                       Modules
# =============================================================================
# third party modules
from mfpml.problems.mf_functions import (Forrester_1a, Forrester_1b,
                                         Forrester_1c, MultiFidelityFunctions,
                                         mf_Bohachevsky, mf_Booth, mf_Borehole,
                                         mf_CurrinExp, mf_Hartman3,
                                         mf_Hartman6, mf_Himmelblau,
                                         mf_Park91A, mf_Park91B, mf_Sixhump)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


def register_problem(problem_name: str) -> MultiFidelityFunctions:
    """register problem according to problem name

    Parameters
    ----------
    problem_name: str
        problem name
    """

    problem_sets = [(Forrester_1a, 'Forrester_1a'),
                    (Forrester_1b, 'Forrester_1b'),
                    (Forrester_1c, 'Forrester_1c'),
                    (mf_Bohachevsky, 'mf_Bohachevsky'),
                    (mf_Booth, 'mf_Booth'),
                    (mf_Borehole, 'mf_Borehole'),
                    (mf_CurrinExp, 'mf_CurrinExp'),
                    (mf_Hartman3, 'mf_Hartman3'),
                    (mf_Hartman6, 'mf_Hartman6'),
                    (mf_Himmelblau, 'mf_Himmelblau'),
                    (mf_Park91A, 'mf_Park91A'),
                    (mf_Park91B, 'mf_Park91B'),
                    (mf_Sixhump, 'mf_Sixhump')]

    try:
        # return  problem according to problem name
        return [problem for problem in problem_sets
                if problem[1] == problem_name][0][0]()
    except IndexError:
        raise ValueError('problem name is not valid')
