from typing import Callable

import numpy as np


def trapezoidal(f: Callable, a: float, b: float, num_intervals: int):

    intervals = np.linspace(a, b, num_intervals + 1)
    delta_x = (b - a) / num_intervals

    sum_f = 0
    for x_i in intervals:
        if x_i in [intervals[0], intervals[-1]]:
            sum_f += f(x_i)
        else:
            sum_f += 2 * f(x_i)

    return delta_x / 2 * sum_f


# if __name__ == '__main__':
#
#     def f(x):
#         return x**2
#
#     result = trapezoidal(f, 0, 5, 20)
#     print(result)
