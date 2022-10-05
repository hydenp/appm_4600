import numpy as np

from rootfinder import lazy_newton_nd
from rootfinder.broyden_nd import broyden_nd
from rootfinder.newton_nd import newton_nd


def print_results(method, r, itrs, error_code):
    print('-' * 15)
    print(method)
    print('root: ', r)
    print('iterations: ', itrs)
    print('ier: ', error_code)


def run_helper(initial_guess):
    print('#' * 30)
    print("INITIAL GUESS: ", initial_guess)

    try:
        root, err_code, iterations = lazy_newton_nd(F_FUNCS, JACOBIAN_FUNCS, initial_guess, TOLERANCE, MAX_ITERATIONS)
        print_results('lazy newton', root, iterations, err_code)
    except Exception as e:
        print('lazy newton failed: ', e)

    try:
        root, err_code, iterations = broyden_nd(F_FUNCS, JACOBIAN_FUNCS, initial_guess, TOLERANCE, MAX_ITERATIONS)
        print_results('broyden', root, iterations, err_code)
    except Exception as e:
        print('broyden failed: ', e)

    try:
        root, err_code, iterations = newton_nd(F_FUNCS, JACOBIAN_FUNCS, initial_guess, TOLERANCE, MAX_ITERATIONS)
        print_results('regular newton', root, iterations, err_code)
    except Exception as e:
        print('regular newton failed: ', e)


if __name__ == '__main__':
    F_FUNCS = [
        [lambda x: x[0] ** 2 + x[1] ** 2 - 4],
        [lambda x: np.exp(x[0]) + x[1] - 1]
    ]

    JACOBIAN_FUNCS = [
        [lambda x: 2 * x[0], lambda x: 2 * x[1]],
        [lambda x: np.exp(x[0]), lambda x: 1]
    ]

    TOLERANCE = 1e-12
    MAX_ITERATIONS = 100

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
