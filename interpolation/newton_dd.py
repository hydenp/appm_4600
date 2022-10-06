from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def evaluate_divide_difference_polynomial(x, x_int, y, N):
    """"""

    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N + 1)

    ptmp[0] = 1.
    for j in range(N):
        ptmp[j + 1] = ptmp[j] * (x - x_int[j])

    '''evaluate the divided difference polynomial'''
    yeval = 0.0
    for j in range(N + 1):
        yeval = yeval + y[0][j] * ptmp[j]

    return yeval


def divided_difference_table(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]))
    return y


def newton_dd(f: Callable, polynomial_degree: int, a: float, b: float, num_sample: int, plot_save_show: str = None):
    """
    :param f:
    :param polynomial_degree:
    :param a: start of the interval
    :param b: end of the interval
    :param num_sample: the number of sample points to test against
    :param plot_save_show: whether to show or save the plot - default neither
    :return:
    """

    ''' create equal-spaced interpolation nodes'''
    x_int = np.linspace(a, b, polynomial_degree + 1)

    ''' create interpolation data'''
    y_int = np.zeros(polynomial_degree + 1)
    for jj in range(polynomial_degree + 1):
        y_int[jj] = f(x_int[jj])

    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros((polynomial_degree + 1, polynomial_degree + 1))

    for j in range(polynomial_degree + 1):
        y[j][0] = y_int[j]

    y = divided_difference_table(x_int, y, polynomial_degree + 1)

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    x_with_num_sample = np.linspace(a, b, num_sample + 1)
    y_with_num_sample = np.zeros(num_sample + 1)
    for kk in range(num_sample + 1):
        y_with_num_sample[kk] = evaluate_divide_difference_polynomial(x_with_num_sample[kk], x_int, y, polynomial_degree)

    ''' create vector with exact values'''
    fex = np.zeros(num_sample + 1)
    for kk in range(num_sample + 1):
        fex[kk] = f(x_with_num_sample[kk])

    if plot_save_show is not None:
        plt.plot(x_with_num_sample, fex)
        plt.plot(x_with_num_sample, y_with_num_sample)
        if plot_save_show == 'save':
            plt.savefig(f'./plots/lab7/newton_dd_{polynomial_degree}.png')
            plt.cla()
        else:
            plt.show()

        if plot_save_show == 'save':
            err = abs(y_with_num_sample - fex)
            plt.plot(x_with_num_sample, err)
            plt.savefig(f'./plots/lab7/newton_dd_error_{polynomial_degree}.png')
            plt.cla()
        else:
            plt.show()
