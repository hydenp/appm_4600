from typing import Callable

import numpy as np
import scipy
from matplotlib import pyplot as plt


def composite_trapezoidal(f: Callable, a: float, b: float, num_intervals: int):
    intervals = np.linspace(a, b, num_intervals + 1)
    delta_x = (b - a) / num_intervals

    sum_f = 0
    for x_i in intervals:
        if x_i in [intervals[0], intervals[-1]]:
            sum_f += f(x_i)
        else:
            sum_f += 2 * f(x_i)

    return delta_x / 2 * sum_f


if __name__ == '__main__':

    def gam(t, x):
        return np.power(t, x - 1) * np.exp(-t)


    x = [2, 4, 6, 8, 10]
    exact = np.array([scipy.special.gamma(x_i) for x_i in x])

    A = 0
    B = 100
    N = 500

    figure, ax = plt.subplots(5, 1)
    figure.set_size_inches(10, 15)
    for i, B in enumerate([50, 100, 300, 500, 1000]):
        for N in [10, 100, 300, 500, 1_000]:
            trapezoid_results = []
            for x_i in x:
                trapezoid_results.append(
                    composite_trapezoidal(lambda t: gam(t, x_i), A, B, N)
                )
            errors = abs(np.array(trapezoid_results) - exact)

            # print(f'B = {B}, N = {N}', errors)
            ax[i].semilogy(x, errors, label=f'N = {N}')
            ax[i].set_title(f'errors with A={A} B={B}')
        ax[i].legend(loc='upper left')

    plt.savefig(f'plots.png')

    # part C
    w, xl = np.polynomial.laguerre.laggauss(100)
    print(np.dot(w, xl))
    # output
    # 1.0000000000003781

