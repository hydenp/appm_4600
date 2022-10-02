import numpy as np

from rootfinder import lazy_newton_nd
from rootfinder import quasi_newton_nd
from rootfinder.newton_nd import newton_nd

F_funcs = [
    [lambda x: x[0] ** 2 + x[1] ** 2 - 4],
    [lambda x: np.exp(x[0]) + x[1] - 1]
]

JACOBIAN_FUNCS = [
    [lambda x: 2 * x[0], lambda x: 2 * x[1]],
    [lambda x: np.exp(x[0]), lambda x: 1]
]

TOLERANCE = 10e-10
MAX_ITERATIONS = 100


def print_results(method, r, itrs, error_code):
    print('-'*15)
    print(method)
    print('root: ', r)
    print('iterations: ', itrs)
    print('ier: ', error_code)


def run_helper(initial_guess):
    root, err_code, iterations = lazy_newton_nd(F_funcs, JACOBIAN_FUNCS, x_0_i, TOLERANCE, MAX_ITERATIONS)
    print_results('lazy newton', root, iterations, err_code)

    root, err_code, iterations = quasi_newton_nd(F_funcs, JACOBIAN_FUNCS, x_0_i, TOLERANCE, MAX_ITERATIONS)
    print_results('quasi newton', root, iterations, err_code)

    root, err_code, iterations = newton_nd(F_funcs, JACOBIAN_FUNCS, x_0_i, TOLERANCE, MAX_ITERATIONS)
    print_results('regular newton',root, iterations, err_code)


if __name__ == '__main__':
    # i
    x_0_i = [1, 1]
    run_helper(x_0_i)
    print('\n')

    # ii
    x_0_ii = [1, -1]
    run_helper(x_0_ii)
    print('\n')

    # iii
    x_0_iii = [0, 0]
    run_helper(x_0_iii)
