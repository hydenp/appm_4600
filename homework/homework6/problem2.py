import math

import numpy as np

from rootfinder.steepest_decent_nd import steepest_descent_nd

if __name__ == '__main__':

    # F_FUNCS = [
    #     [lambda x: x[0] + math.cos(x[0] * x[1] * x[2]) - 1],
    #     [lambda x: pow((1 - x[0]), 0.25) + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1],
    #     [lambda x: -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1]
    # ]
    #
    # JACOBIAN_FUNCS = [
    #     [lambda x: 1 - x[1] * x[2] * math.sin(x[0] * x[1] * x[2]), lambda x: - x[0] * x[2] * math.sin(x[0] * x[1] * x[2]),
    #      lambda x: - x[0] * x[1] * math.sin(x[0] * x[1] * x[2])],
    #     [lambda x: -0.25 * pow((1 - x[0]), -0.75), lambda x: 1, lambda x: 2 * 0.05 * x[2] - 0.15],
    #     [lambda x: -2 * x[0], lambda x: 2 * 0.1 * x[1] + 0.01, lambda x: 1]
    # ]
    #
    # TOLERANCE = 10e-6
    # MAX_ITERATIONS = 20
    # root, exit_code, iters = newton_nd(F_FUNCS, JACOBIAN_FUNCS, [4, 1, 1], TOLERANCE, MAX_ITERATIONS)

    # STEEPEST DESCENT

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
