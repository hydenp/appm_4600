from typing import Callable

import numpy as np


def lazy_newton_nd(
        f_at: Callable,
        jacobian_at: Callable,
        x0: [float],
        tolerance: float,
        max_iterations: int) -> ([float], int, int):
    """
    :param f_at: the function at each step
    :param jacobian_at: a 2d matrix of lambdas that take in one list which is the point at which to evaluate
    :param x0: initial guess
    :param tolerance
    :param max_iterations
    :return: root approximation, error code, number of iterations

    assumes each function takes in a single array argument for the point to evaluate at
    assumes each jacobian function takes in a single array argument for the point to evaluate at
    """
    x1 = None
    inverse_jacobian = np.linalg.inv(jacobian_at(x0))

    for iteration in range(max_iterations):

        # evaluate the next iteration with reshaped input
        f_at_x0 = f_at(x0)

        x1 = np.array(x0) - inverse_jacobian @ f_at_x0

        # check the tolerance has been met
        if np.linalg.norm(x1 - x0) < tolerance:
            # x_star = x1
            # ier = 0
            return x1, 0, iteration

        x0 = x1

    # x_star = x1
    # ier = 1
    return x1, 1, max_iterations
