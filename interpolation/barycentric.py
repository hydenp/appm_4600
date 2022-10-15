from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def eval_barycentric(x: float, x_interpolation_pts: [], y_interpolation_pts: [], N, weights: []):
    # check if the exact points exists
    if x in x_interpolation_pts:
        i = list(x_interpolation_pts).index(x)
        return y_interpolation_pts[i]

    top_sum = 0
    bottom_sum = 0
    for i in range(N + 1):
        top_sum += weights[i] / (x - x_interpolation_pts[i]) * y_interpolation_pts[i]
        bottom_sum += weights[i] / (x - x_interpolation_pts[i])

    return top_sum / bottom_sum


def calculate_weights(x_interpolation_nodes: []):
    weights = []
    for i in range(len(x_interpolation_nodes)):
        w_j = 1
        for j in range(len(x_interpolation_nodes)):
            if x_interpolation_nodes[i] != x_interpolation_nodes[j]:
                w_j *= 1 / (x_interpolation_nodes[i] - x_interpolation_nodes[j])
        weights.append(w_j)

    return weights


def barycentric(f: Callable, num_interpolation_nodes: int, a: int, b: int, num_sample_points: int,
                file_prefix: str = '',
                special_x: [] = None):

    if special_x is None:
        # create equal-spaced interpolation nodes
        x_interpolation_pts = np.linspace(a, b, num_interpolation_nodes + 1)
    else:
        x_interpolation_pts = special_x

    # create interpolation data
    y_interpolation_pts = np.zeros(num_interpolation_nodes + 1)
    for j in range(num_interpolation_nodes + 1):
        y_interpolation_pts[j] = f(x_interpolation_pts[j])

    weights = calculate_weights(x_interpolation_pts)

    # create points for evaluating the Lagrange interpolating polynomial
    x_interpolation_pts_eval = np.linspace(a, b, num_sample_points + 1)
    y_interpolation_pts_eval = np.zeros(num_sample_points + 1)
    for i in range(num_sample_points + 1):
        y_interpolation_pts_eval[i] = eval_barycentric(x_interpolation_pts_eval[i], x_interpolation_pts,
                                                       y_interpolation_pts,
                                                       num_interpolation_nodes, weights)

    # create vector with exact values
    f_exact = np.zeros(num_sample_points + 1)
    for i in range(num_sample_points + 1):
        f_exact[i] = f(x_interpolation_pts_eval[i])

    _, ax = plt.subplots(1, 2)
    # plotting exact and interpolation
    ax[0].plot(x_interpolation_pts_eval, f_exact, label='Exact')
    ax[0].plot(x_interpolation_pts_eval, y_interpolation_pts_eval, 'rx', label='barycentric')
    ax[0].set_title(f'Exact and Interpolated Plot with N={num_interpolation_nodes}')

    # plotting error
    errors = abs(y_interpolation_pts_eval - f_exact)
    ax[1].semilogy(x_interpolation_pts_eval, errors)
    ax[1].set_title('Error')

    plt.savefig(f'{file_prefix}barycentric-n-{num_interpolation_nodes}.png')
    return y_interpolation_pts_eval, errors
