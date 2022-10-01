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

    count = 0
    while count < max_iterations:
        count = count + 1
        x1 = f(x0)
        approximations[count] = x1
        if abs(x1 - x0) < tol:
            # x_star = x1
            # ier = 0
            # approximations = approximations[:count]
            return x1, approximations[:count], 0
        x0 = x1

    # x_star = x1
    # ier = 1
    return x1, approximations, 1
