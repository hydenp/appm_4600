import math

import numpy as np


def evaluate_f(x):
    return np.array([
        [4 * x[0] ** 2 + x[1] ** 2 - 4],
        [x[0] - x[1] - math.sin(x[0] - x[1])]]
    )


def evaluate_j(x: list):
    return np.array([[4 * x[0], 2 * x[1]],
                     [1 - math.cos(x[0] - x[1]), 1 + math.cos(x[0] - x[1])]])


def slacker_newton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    x_n = x0

    jacobian = evaluate_j(x_n)
    inverse_jacobian = np.linalg.inv(jacobian)

    iterate_difference = float('-inf')  # difference between current guess

    for its in range(Nmax):

        f_at_x_n = evaluate_f(x_n)

        x_n_plus_1 = x_n - inverse_jacobian.dot(f_at_x_n)

        curr_diff = sum([x_n_plus_1[0] - x_n[0], x_n_plus_1[1] - x_n[1]])/2
        if np.linalg.norm(x_n_plus_1 - x_n) < tol:
            x_star = x_n_plus_1
            ier = 0
            return [x_star, ier, its]

        # in lazy newton, check last error
        if abs(curr_diff) > iterate_difference:
            inverse_jacobian = np.linalg.inv(jacobian)

        iterate_difference = curr_diff

        x_n = x_n_plus_1

    x_star = x_n_plus_1
    ier = 1
    return [x_star, ier, its]


if __name__ == '__main__':

    x_0 = [1, 0]
    epsilon = 10e-10

    root, ier, iterations = slacker_newton(x_0, epsilon, 100)
    print(f'root: {root}\nier: {ier}, iterations: {iterations}')
