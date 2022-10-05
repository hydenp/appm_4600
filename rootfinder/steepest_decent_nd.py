import math
from typing import Callable

import numpy as np


def evaluate_jacobian(jacobian_functions: [[Callable]], x: [float]):
    jacobian = np.eye(len(jacobian_functions), len(jacobian_functions[0]))
    for row_index, row in enumerate(jacobian_functions):
        for col_index, function in enumerate(row):
            jacobian[row_index][col_index] = function(x)

    return jacobian


def steepest_descent_nd(f: Callable, jacobian: [[Callable]], x_0: [float], tolerance: float, max_iterations: int):
    x_n = x_0

    def g_of(x: [int]) -> float:
        return sum(
            i ** 2 for i in f(x)
        )

    def z_of(x: [int]) -> np.asarray:
        return np.asarray(jacobian(x).T @ f(x))

    # STEP 2
    for iteration in range(max_iterations):

        # STEP 3
        g_1 = g_of(x_n)
        z = z_of(x_n)

        # STEP 4
        # check for zero gradient
        z_0 = np.linalg.norm(z)
        if z_0 == 0:
            return x_n, 1, iteration

        # STEP 5
        # make z unit
        z = [z_i / z_0 for z_i in z]
        a_1 = 0
        a_3 = 1

        # calc x - a3*z
        # print(np.array(x_n) - np.array(z))
        g_3 = g_of(np.array(x_n) - a_3 * np.array(z))

        # STEP 6
        while g_3 >= g_1:

            # STEP 7
            a_3 = a_3 / 2
            g_3 = g_of(np.asarray(x_n) - (a_3 * np.asarray(z)))

            # STEP 8
            # check if local minima and no likely improvement
            if a_3 < tolerance / 2:
                return x_n, 1, iteration

        # STEP 9
        a_2 = a_3 / 2
        g_2 = g_of(np.array(x_n) - a_2 * np.array(z))

        # STEP 10
        h_1 = (g_2 - g_1) / a_2
        h_2 = (g_3 - g_2) / (a_3 - a_2)
        h_3 = (h_2 - h_1) / a_3

        # STEP 11
        a_0 = 0.5 * (a_2 - h_1 / h_3)
        g_0 = g_of(np.asarray(x_n) - a_0 * np.asarray(z))

        # STEP 12
        if g_0 < g_3:
            g_min = g_0
            a = a_0
        else:
            g_min = g_3
            a = a_3

        # STEP 13
        x_n = np.array(x_n) - a * np.array(z)

        # STEP 14
        if abs(g_min - g_1) < tolerance:
            return x_n, 0, iteration

    # STEP 16
    return x_n, 2, max_iterations


if __name__ == '__main__':
    F_FUNCS = lambda x: [
        x[0] + math.cos(x[0] * x[1] * x[2]) - 1,
        pow((1 - x[0]), 0.25) + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1,
        -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1
    ]

    JACOBIAN = lambda x: np.asarray([
        [1 - x[1] * x[2] * math.sin(x[0] * x[1] * x[2]),
         - x[0] * x[2] * math.sin(x[0] * x[1] * x[2]),
         - x[0] * x[1] * math.sin(x[0] * x[1] * x[2])],
        [-1 * 0.25 * pow((1 - x[0]), -0.75), 1, 2 * 0.05 * x[2] - 0.15],
        [-2 * x[0], -0.1 * 2 * x[1], 1],
    ])

    tolerance = 10e-13
    max_iterations = 100
    x_0 = [0, -2, -2]

    print(F_FUNCS([1, 2, 3]))

    # steepest_descent_nd(F_FUNCS, JACOBIAN_FUNCTIONS, x_0, tolerance, max_iterations)
    result = steepest_descent_nd(F_FUNCS, JACOBIAN, x_0, tolerance, max_iterations)
    print(result)
    print()
    res_at_f = F_FUNCS(result[0])
    for t in res_at_f:
        print(round(t, 6))


# def steepest_descent_nd_2(functions: [[Callable]], jacobian_functions: [[Callable]], x0: [int], tolerance: float,
#                           max_iterations: int):
#     # check if x0 is our answer
#     f_at_x0 = evaluate_f(functions, x0)
#     if np.linalg.norm(f_at_x0) == 0:
#         return x0, 0, 0
#
#     x_k = x0
#
#     # iterate
#     for iteration in range(max_iterations):
#
#         # TODO: make sure dimensions correct, calculate beta
#         jacobian = evaluate_jacobian(jacobian_functions, x_k)
#         x_k_plus_1 = x_k - 2.0 * beta * jacobian.transpose() * evaluate_f(functions, x_k)
#
#         # check if root/minima is found
#         if np.linalg.norm(x_k_plus_1) < tolerance:
#             return x_k_plus_1, 0, iteration
#
#         x_k = x_k_plus_1
#
#     # root not found after max iterations
#     return x_k, 1, max_iterations
