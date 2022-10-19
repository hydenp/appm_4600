from typing import Callable

import numpy as np


def eval_lagrange(x, x_int_pts, y_int_pts):

    lj = np.ones(len(x_int_pts))
    for count in range(len(x_int_pts)):
        for j in range(len(x_int_pts)):
            if x_int_pts[count] != x_int_pts[j]:
                lj[count] = lj[count] * (x - x_int_pts[j]) / \
                            (x_int_pts[count] - x_int_pts[j])

    y_evaluation = 0.0
    for j in range(len(x_int_pts)):
        y_evaluation = y_evaluation + y_int_pts[j] * lj[j]

    return y_evaluation


def lagrange(f: Callable, x_int_pts, x_eval_pts):
    """
    :param f:
    :param x_int_pts:
    :param x_eval_pts
    :return:
    """

    # create interpolation data
    y_interpolation_nodes = np.zeros(len(x_int_pts))
    for i, x in enumerate(x_int_pts):
        y_interpolation_nodes[i] = f(x)

    # create points for evaluating the Lagrange interpolating polynomial
    y_eval_pts = np.zeros(len(x_eval_pts))

    for kk in range(len(x_eval_pts)):
        y_eval_pts[kk] = eval_lagrange(x_eval_pts[kk], x_int_pts, y_interpolation_nodes)

    return y_eval_pts
