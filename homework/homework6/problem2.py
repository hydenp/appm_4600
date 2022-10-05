import math

import numpy as np

from rootfinder.newton_nd import newton_nd
from rootfinder.steepest_decent_nd import steepest_descent_nd


def print_results(method, init_guess, found_root, function_f, itrs, err_code):
    print('-' * 15)
    print('method: ', method)
    print('initial guess: ', init_guess)
    print('root: ', found_root)
    print('f at root: ', function_f(found_root))
    print('iterations: ', itrs)
    print('ier: ', err_code)


if __name__ == '__main__':
    # FUNCTIONS
    def f(x):
        return [
            x[0] + math.cos(x[0] * x[1] * x[2]) - 1,
            pow((1 - x[0]), 0.25) + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1,
            -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1
        ]


    def j(x):
        return np.array([
            [1 - x[1] * x[2] * math.sin(x[0] * x[1] * x[2]),
             - x[0] * x[2] * math.sin(x[0] * x[1] * x[2]),
             - x[0] * x[1] * math.sin(x[0] * x[1] * x[2])],
            [-1 * 0.25 * pow((1 - x[0]), -0.75), 1, 2 * 0.05 * x[2] - 0.15],
            [-2 * x[0], -0.1 * 2 * x[1], 1],
        ])

    tolerance = 10e-13
    max_iterations = 100

    # NEWTON
    initial_guess = [0, -2, -2]
    root, error_code, iterations = newton_nd(f, j, initial_guess, tolerance, max_iterations)
    print_results('newton', initial_guess, root, f, iterations, error_code)

    print()

    # STEEPEST DESCENT
    root, error_code, iterations = steepest_descent_nd(f, j, initial_guess, tolerance, max_iterations)
    print_results('steepest descent', initial_guess, root, f, iterations, error_code)

    print()

    # NEWTON with the STEEPEST DESCENT guess
    initial_guess = [0, 1, 3]
    tolerance = 5e-2
    root_g, error_code, iterations = steepest_descent_nd(f, j, initial_guess, tolerance, max_iterations)
    print_results('steepest descent for newton initial guess', initial_guess, root, f, iterations, error_code)
    root, error_code, iterations = newton_nd(f, j, root_g, tolerance, max_iterations)
    print_results('newton with steepest descent as initial guess', root_g, root, f, iterations, error_code)
