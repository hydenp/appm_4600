from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def hermite(f: Callable, fp: Callable, x_int_pts: [float], x_eval_pts: [float], file_prefix: str = ''):
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
        f_q = sum(
            y_int_nodes[j] * Q(x, j) for j in range(len(x_int_pts))
        )
        fp_r = sum(
            yp_int_nodes[j] * R(x, j) for j in range(len(x_int_pts))
        )
        y_hermite_nodes.append(f_q + fp_r)

    # create vector with exact values
    f_exact = np.ones(len(x_eval_pts))
    for index, x in enumerate(x_eval_pts):
        f_exact[index] = f(x)

    _, ax = plt.subplots(1, 2)
    # plotting exact and interpolation
    ax[0].plot(x_eval_pts, f_exact, label='Exact')
    ax[0].plot(x_eval_pts, y_hermite_nodes, 'r-', label='hermite')
    ax[0].set_title(f'Exact and Interpolated Plot with N={len(x_int_pts)}')

    # plotting error
    errors = abs(y_hermite_nodes - np.array(f_exact))
    ax[1].semilogy(x_eval_pts, errors)
    ax[1].set_title('Error')
    plt.savefig(f'{file_prefix}hermite-n-{len(x_int_pts)}.png')

    return y_hermite_nodes, errors
