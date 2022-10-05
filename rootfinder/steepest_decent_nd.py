from typing import Callable

import numpy as np


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
