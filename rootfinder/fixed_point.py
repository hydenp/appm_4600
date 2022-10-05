from typing import Callable

import numpy as np


def fixed_point(f: Callable, x0: float, tol: float, max_iterations: int):
    """
    :param f function we want to find root of
    :param x0  initial guess
    :param tol - stopping tolerance
    :param max_iterations - max number of iterations
    """
    x1 = None
    approximations = np.zeros((max_iterations + 1, 1))

    iteration = 0
    while iteration < max_iterations:
        iteration = iteration + 1
        x1 = f(x0)
        approximations[iteration] = x1
        if abs(x1 - x0) < tol:
            # x_star = x1
            # ier = 0
            # approximations = approximations[:iteration]
            return x1, 0, iteration+1, approximations[:iteration]
        x0 = x1

    # x_star = x1
    # ier = 1
    return x1, 0, max_iterations, approximations
