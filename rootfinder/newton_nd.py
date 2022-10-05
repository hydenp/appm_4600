from typing import Callable

import numpy as np


def newton_nd(
        f_at: Callable,
        jacobian_at: Callable,
        x0: [float],
        tolerance: float,
        max_iterations: int) -> ([float], int, int):
    """
    :param f_at: the function at each step
    :param jacobian_at: a 2d matrix of lambdas that take in one list which is the point at which to evaluate
    :param x0:
    :param tolerance:
    :param max_iterations: max numb
    :return: x_star= approx root, ier = error message, its = num its

    assumes each function takes in a single array argument for the point to evaluate at
    assumes each jacobian function takes in a single array argument for the point to evaluate at
    """

    x1 = None

    for iteration in range(max_iterations):

        # inverse_jacobian = evaluate_jacobian_inverse(jacobian_at, x0)
        inverse_jacobian = np.linalg.inv(jacobian_at(x0))

        # evaluate the next iteration with reshaped input
        f_at_x0 = f_at(x0)

        x1 = np.array(x0) - inverse_jacobian @ f_at_x0

        # check the tolerance has been met
        if np.linalg.norm(x1 - x0) < tolerance:
            # x_star = x1
            # ier = 0
            return x1, 0, iteration + 1

        x0 = x1

    # x_star = x1
    # ier = 1
    return x1, 1, max_iterations
