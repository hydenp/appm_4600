from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def create_vandermonde(num_interpolation_nodes):

    x = np.linspace(-1, 1, num_interpolation_nodes + 1)
    vandermonde = np.zeros((num_interpolation_nodes + 1, num_interpolation_nodes + 1))
    for j in range(0, num_interpolation_nodes + 1):
        vandermonde[:, j] = x ** j

    return vandermonde, x


def monomial_basis(f: Callable, num_interpolation_nodes: int, num_sample_pts: int,
                   plot_save_show: str = None, file_prefix: str = ''):

    # Vandermonde matrix and interpolation points
    vandermonde, x = create_vandermonde(num_interpolation_nodes)

    # sample the function for interpolation nodes
    f_at_x = f(x)

    # solve for the coefficients of the monomials Vc = F
    coefficients = np.linalg.solve(vandermonde, f_at_x)

    # sample the function on finer grid to compare errors
    x_with_num_sample = np.linspace(-1, 1, num_sample_pts)
    v_with_num_sample = np.zeros((num_sample_pts, num_interpolation_nodes + 1))
    for j in range(0, num_interpolation_nodes + 1):
        v_with_num_sample[:, j] = x_with_num_sample ** j
    f_with_num_sample = f(x_with_num_sample)

    # calculate the error
    error = np.abs(v_with_num_sample @ coefficients - f_with_num_sample)

    if plot_save_show is not None:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(x_with_num_sample, f_with_num_sample, 'o-')
        ax[0].plot(x_with_num_sample, v_with_num_sample @ coefficients, 'b--')
        ax[0].set_title('interpolation and function')
        ax[1].semilogy(x_with_num_sample, error, 'r-')
        ax[1].set_title('absolute error')
        if plot_save_show == 'save':
            plt.savefig(f'{file_prefix}monomial-degree-n-{num_interpolation_nodes}.png')
            plt.cla()
        else:
            plt.show()

    return coefficients
