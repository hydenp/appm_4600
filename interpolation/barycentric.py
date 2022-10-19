from typing import Callable

import numpy as np


def eval_barycentric(x: float, x_int_pts: [], y_int_pts: [], N, weights: []):
    # check if the exact points exists
    if x in x_int_pts:
        i = list(x_int_pts).index(x)
        return y_int_pts[i]

    top_sum = 0
    bottom_sum = 0
    for i, x_i in enumerate(x_int_pts):
        top_sum += weights[i] / (x - x_i) * y_int_pts[i]
        bottom_sum += weights[i] / (x - x_i)

    return top_sum / bottom_sum


def calculate_weights(x_int_pts: []):
    weights = []
    for i in range(len(x_int_pts)):
        w_j = 1
        for j in range(len(x_int_pts)):
            if x_int_pts[i] != x_int_pts[j]:
                w_j *= 1 / (x_int_pts[i] - x_int_pts[j])
        weights.append(w_j)

    return weights


def barycentric(f: Callable, x_int_pts: np.array, x_eval_pts):
    # create interpolation data
    y_int_pts = np.zeros(len(x_int_pts))
    for i, x in enumerate(x_int_pts):
        y_int_pts[i] = f(x)

    weights = calculate_weights(x_int_pts)

    # create points for evaluating the Lagrange interpolating polynomial
    y_eval_pts = np.zeros(len(x_eval_pts))
    for i, x in enumerate(x_eval_pts):
        y_eval_pts[i] = eval_barycentric(x, x_int_pts,
                                         y_int_pts,
                                         len(y_int_pts), weights)

    return y_eval_pts
