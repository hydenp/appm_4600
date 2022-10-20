from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def get_sub_interval_indexes(a: float, b: float, points: []) -> []:
    above = np.where(points >= a)[0]
    below = np.where(points <= b)[0]
    return list(set(above) & set(below))


def cubic_spline(f: Callable, x_int_pts: np.array, x_eval_pts, N) -> np.array:
    # following the algo from the book

    y_int_pts = np.array([f(x) for x in x_int_pts])
    h = np.diff(x_int_pts)

    # construct A
    A = np.zeros((N+1, N+1))
    A[0][0] = 1
    A[N][N] = 1
    for j in range(N-1):
        A[j+1][j] = h[j-1]
        A[j+1][j+1] = 2 * (h[j-1] + h[j])
        A[j+1][j+2] = h[j]
    
    print(A)

    # Step Two
    a = np.array([f(x) for x in x_int_pts])
    # for i in range(1, N - 1):
    #     a[i] = (3 / h[i]) * (y_int_pts[i + 1] - y_int_pts[i]) - \
    #         (3 / h[i - 1]) * (y_int_pts[i] - y_int_pts[i - 1])

    b = np.zeros((N+1, 1))
    for j in range(1, N-1):
        b[j][0] = (3/h[j])*(a[j+1] - a[j]) - (3/h[j-1]) * (a[j] - a[j-1])
    print(b)

    c = np.linalg.solve(A, b)
    c = np.reshape(c, (1, N+1))[0]

    print(c)

    # Step 5
    b = np.ones(N)
    d = np.ones(N)
    for j in range(N - 2, -1, -1):
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # Added step
    # evaluate for all the x points
    y_eval_pts = []
    for x in x_eval_pts:

        x_j = 1
        # pick x_j as the beginning point in the interval

        y_eval_pts.append(
            a[j] + b[j] *(x - x_j) + c[j]*(x - x_j)**2 + d[j]*(x - x_j)**2
        )

    return y_eval_pts


# def cubic_spline_driver(f: Callable, x_int_pts: np.array, x_eval_pts, N, num_intervals):
#
#     y_eval_pts = []
#     x_intervals = np.linspace(x_eval_pts[0], x_eval_pts[-1], num_intervals + 1)
#
#     # evaluate each piecewise potion
#     for i in range(num_intervals):
#         interval_indices = get_sub_interval_indexes(x_intervals[i], x_intervals[i + 1], x_eval_pts)
#
#         interval_pts = [x_eval_pts[ind] for ind in interval_indices]
#
#         y_eval_pts.extend(
#             cubic_spline(f, x_int_pts, interval_pts, N)
#         )
#
#     return y_eval_pts


if __name__ == '__main__':

    def plot_int(x_eval_pts, f_exact: np.array, y_eval_pts: np.array, N: int, save=False, file_prefix='') -> None:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12, 7)

        # plot interpolation with exact
        ax[0].plot(x_eval_pts, f_exact, label='exact')
        ax[0].plot(x_eval_pts, y_eval_pts, 'r--', label='interpolation')
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'Interpolation with N={N}')

        # plotting error
        err = abs(y_eval_pts - f_exact)
        ax[1].set_title('Error')
        ax[1].plot(x_eval_pts, err)

        if save:
            plt.savefig(f'{file_prefix}-n-{N}')
            plt.cla()
        else:
            plt.show()

    # function to interpolate

    def f(x):
        return 1 / (1 + x ** 2)

    def fp(x):
        return -2 * x / (1 + x ** 2) ** 2

    N = 5
    x_eval_pts = np.linspace(-5, 5, 50)
    x_int_pts = np.linspace(-5, 5, N)
    f_exact = [f(x) for x in x_eval_pts]

    cubic_pts = cubic_spline(f, x_int_pts, x_eval_pts, N)

    plot_int(x_eval_pts, f_exact, np.array(cubic_pts), N)





    # # Step Three
    # l = np.ones(N)
    # l[0] = 1
    # mu = np.ones(N)
    # mu[0] = 0
    # z = np.ones(N)
    # z[0] = 0

    # # Step Four
    # for i in range(1, N - 1):
    #     l[i] = 2 * (x_int_pts[i + 1] - x_int_pts[i - 1]) - h[i - 1] * mu[i - 1]
    #     mu[i] = h[i] / l[i]
    #     z[i] = (a[i] - h[i - 1] * z[i - 1]) / l[i]

    # # Step 5
    # l[-1] = 1
    # z[-1] = 0
    # c = np.ones(N)
    # b = np.ones(N)
    # d = np.ones(N)
    # c[-1] = 0

    # # Step 6
    # for j in range(N - 2, -1, -1):
    #     c[j] = z[j] - mu[j] * c[j + 1]
    #     b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
    #     d[j] = (c[j + 1] - c[j]) / (3 * h[j])
