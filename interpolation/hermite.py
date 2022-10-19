from typing import Callable

import numpy as np


def hermite(f: Callable, fp: Callable, x_int_pts: [float], x_eval_pts: [float]) -> np.array:
    y_int_nodes = [f(x) for x in x_int_pts]
    yp_int_nodes = [fp(x) for x in x_int_pts]

    def lp(j):
        return 1.0 / np.prod([
            x_int_pts[j] - x_j if x_int_pts[j] != x_j else 1
            for x_j in x_int_pts
        ])

    def l(x: float, j: int):
        return np.prod([
            (x - x_int_pts[i]) / (x_int_pts[j] - x_int_pts[i]) if i != j else 1
            for i in range(len(x_int_pts))
        ])

    def Q(x, j):
        return (1.0 - 2.0 * (x - x_int_pts[j]) * lp(j)) * (l(x, j) ** 2)

    def R(x, j):
        return (x - x_int_pts[j]) * (l(x, j) ** 2)

    y_hermite_nodes = []
    for x in x_eval_pts:
        y_hermite_nodes.append(
            sum(
                y_int_nodes[j] * Q(x, j) for j in range(len(x_int_pts))
            ) + sum(
                yp_int_nodes[j] * R(x, j) for j in range(len(x_int_pts))
            )
        )

    return np.array(y_hermite_nodes)
