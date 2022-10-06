from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def monomial_basis(f: Callable, max_polynomial_degree: int, num_sample: int,
                   plot_save_show: str = None):
    # interpolation nodes
    x = np.linspace(-1, 1, max_polynomial_degree + 1)

    # Vandermonde matrix
    vandermonde = np.zeros((max_polynomial_degree + 1, max_polynomial_degree + 1))
    for j in range(0, max_polynomial_degree + 1):
        vandermonde[:, j] = x ** j

    # sample the function for interpolation nodes
    f_at_x = f(x)

    # solve for the coefficients of the monomials Vc = F
    coefficients = np.linalg.solve(vandermonde, f_at_x)

    # sample the function on finer grid to compare errors
    x_with_num_sample = np.linspace(-1, 1, num_sample)
    v_with_num_sample = np.zeros((num_sample, max_polynomial_degree + 1))
    for j in range(0, max_polynomial_degree + 1):
        v_with_num_sample[:, j] = x_with_num_sample ** j
    f_with_num_sample = f(x_with_num_sample)

    # calculate the error
    error = np.abs(v_with_num_sample @ coefficients - f_with_num_sample)

    if plot_save_show is not None:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(x_with_num_sample, f_with_num_sample, 'k-')
        ax[0].plot(x_with_num_sample, v_with_num_sample @ coefficients, 'b--')
        ax[0].set_title('interpolation and function')
        ax[1].semilogy(x_with_num_sample, error, 'r-')
        ax[1].set_title('absolute error')
        if plot_save_show == 'save':
            plt.savefig(f'./plots/lab7/monomial_degree{max_polynomial_degree}.png')
            plt.cla()
        else:
            plt.show()

    return coefficients
