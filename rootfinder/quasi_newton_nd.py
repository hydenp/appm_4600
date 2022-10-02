from typing import Callable

import numpy as np

from rootfinder.newton_nd_helpers import evaluate_jacobian_inverse, evaluate_f


def quasi_newton_nd(
        functions: [[Callable]],
        jacobian_functions: [[Callable]],
        x0: [[int]],
        tolerance,
        max_iterations) -> [[[]], int, int]:
    """
    :param functions: the function at each step
    :param jacobian_functions: a 2d matrix of lambdas that take in one list which is the point at which to evaluate
    :param x0:
    :param tolerance:
    :param max_iterations: max numb
    :return: x_star= approx root, ier = error message, its = num its

    assumes each function takes in a single array argument for the point to evaluate at
    assumes each jacobian function takes in a single array argument for the point to evaluate at
    """
    x1 = None
    last_norm = float('-inf')
    inverse_jacobian = evaluate_jacobian_inverse(jacobian_functions, x0)
    for iteration in range(max_iterations):

        # evaluate the next iteration with reshaped input
        f_at_x0 = evaluate_f(functions, np.reshape(x0, (1, len(x0)))[0])

        # reshape x0 to a column matrix for subtraction step
        x0 = np.reshape(x0, (len(x0), 1))
        x1 = np.subtract(x0, np.matmul(inverse_jacobian, f_at_x0))

        # calculate the norm for tolerance and jacobian recalc test
        norm = np.linalg.norm(x1 - x0)

        # check the tolerance has been met
        if norm < tolerance:
            # x_star = x1
            # ier = 0
            return x1, 0, iteration

        x0 = x1

        # the condition at which we decide to calculate a new jacobian
        if iteration > 0 and norm > last_norm:
            last_norm = norm
            inverse_jacobian = evaluate_jacobian_inverse(jacobian_functions, x0)

    # x_star = x1
    # ier = 1
    return x1, 1, max_iterations
