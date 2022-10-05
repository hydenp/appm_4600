from typing import Callable

import numpy as np


def evaluate_jacobian_inverse(functions: [[Callable]], x: [float]) -> np.array:
    """
    :param functions: matrices of derivatives for jacobian - assumes square
    :param x: the point at which to evaluate the jacobian - assume same length as funcs
    :return: inverse jacobian at x

    Assumes that each function takes in a single array for the point it evaluates at.
    """
    jacobian = np.eye(len(functions), len(functions[0]))

    for row_index, row in enumerate(functions):
        for col_index, function in enumerate(row):
            jacobian[row_index][col_index] = function(x)

    return np.linalg.inv(jacobian)


def evaluate_f(functions: [[Callable]], x: [int]) -> np.array:
    """
    Assumes that each function takes in a single array for the point it evaluates at
    """
    """"
    :param functions: the functions that make up F
    :param x: the point at which to evaluate F
    :return: an array of points for F(x)
    """

    f_at_x = np.eye(len(functions), len(functions[0]))

    for row_index, row in enumerate(functions):
        for col_index, func in enumerate(row):
            f_at_x[row_index][col_index] = func(x)

    return f_at_x
