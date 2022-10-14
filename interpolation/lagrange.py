from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def eval_lagrange(x, x_interpolation_pts, y_interpolation_pts, num_interpolation_nodes):
    lj = np.ones(num_interpolation_nodes + 1)

    for count in range(num_interpolation_nodes + 1):
        for j in range(num_interpolation_nodes + 1):
            if x_interpolation_pts[count] != x_interpolation_pts[j]:
                lj[count] = lj[count] * (x - x_interpolation_pts[j]) / \
                            (x_interpolation_pts[count] - x_interpolation_pts[j])

    y_evaluation = 0.0
    for j in range(num_interpolation_nodes + 1):
        y_evaluation = y_evaluation + y_interpolation_pts[j] * lj[j]

    return y_evaluation


def lagrange(f: Callable, num_interpolation_nodes: int, a: int, b: int, num_sample_pts: int, file_prefix: str = '',
             special_x: [] = None):
    """

    :param f:
    :param num_interpolation_nodes:
    :param a:
    :param b:
    :param num_sample_pts:
    :param file_prefix
    :param special_x
    :return:
    """

    if special_x is None:
        # create equal-spaced interpolation nodes
        x_interpolation_nodes = np.linspace(a, b, num_interpolation_nodes + 1)
    else:
        x_interpolation_nodes = special_x

    # create interpolation data
    y_interpolation_nodes = np.zeros(num_interpolation_nodes + 1)
    for jj in range(num_interpolation_nodes + 1):
        y_interpolation_nodes[jj] = f(x_interpolation_nodes[jj])

    # create points for evaluating the Lagrange interpolating polynomial
    x_evaluation_points = np.linspace(a, b, num_sample_pts + 1)
    y_evaluation_points = np.zeros(num_sample_pts + 1)

    for kk in range(num_sample_pts + 1):
        y_evaluation_points[kk] = eval_lagrange(x_evaluation_points[kk], x_interpolation_nodes, y_interpolation_nodes,
                                                num_interpolation_nodes)

    # create vector with exact values
    f_at_x = np.zeros(num_sample_pts + 1)
    for kk in range(num_sample_pts + 1):
        f_at_x[kk] = f(x_evaluation_points[kk])

    # plot interpolation with exact
    plt.plot(x_evaluation_points, f_at_x, 'o', label='exact')
    plt.plot(x_evaluation_points, y_evaluation_points, 'rx', label='lagrange')
    plt.legend(loc='upper right')
    plt.savefig(f'{file_prefix}lagrange-int-n-{num_interpolation_nodes}')
    plt.cla()

    err = abs(y_evaluation_points - f_at_x)
    plt.title('Error')
    plt.plot(x_evaluation_points, err)
    plt.savefig(f'{file_prefix}lagrange-error-n-{num_interpolation_nodes}')
    plt.cla()

    return err
