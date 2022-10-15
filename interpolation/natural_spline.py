from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def get_sub_interval_indexes(a: float, b: float, points: []) -> []:
    above = np.where(points >= a)[0]
    below = np.where(points <= b)[0]
    return list(set(above) & set(below))


def line_through(x_0: float, x_1: float, f_0: float, f_1: float) -> [(float, float)]:
    return lambda x: (f_1 - f_0) / (x_1 - x_0) * (x - x_0) + f_0


def natural_spline(f: Callable, num_intervals: int, a: float, b: float,
                   num_evaluation_pts: int, file_prefix: str = ''):
    # create the intervals for piecewise approximations
    x_intervals = np.linspace(a, b, num_intervals + 1)

    # create vector to store the evaluation of the linear splines
    y_interpolation_evaluations = np.zeros(num_evaluation_pts)

    # create the points we want to sample at
    x_evaluation_pts = np.linspace(a, b, num_evaluation_pts)

    for i in range(num_intervals):
        # find indices of x_evaluation_pts in interval (x_intervals(jint),x_intervals(jint+1))
        # let ind denote the indices in the intervals
        # let n denote the length of ind
        interval_indices = get_sub_interval_indexes(x_intervals[i], x_intervals[i + 1], x_evaluation_pts)

        # temporarily store your info for creating a line in the interval of interest
        a1 = x_intervals[i]
        fa1 = f(a1)
        b1 = x_intervals[i + 1]
        fb1 = f(b1)

        line = line_through(a1, b1, fa1, fb1)
        for kk in range(len(interval_indices)):
            # use your line evaluator to evaluate the lines at each of the points in the interval
            # yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with the points (a1,fa1) and (b1,fb1)'''

            y_interpolation_evaluations[interval_indices[kk]] = line(x_evaluation_pts[interval_indices[kk]])

    # evaluate f at the evaluation points'''
    f_exact = np.zeros(num_evaluation_pts)
    for j in range(num_evaluation_pts):
        f_exact[j] = f(x_evaluation_pts[j])

    _, ax = plt.subplots(1, 2)
    # plotting exact and interpolation
    ax[0].plot(x_evaluation_pts, f_exact, label='exact')
    ax[0].plot(x_evaluation_pts, y_interpolation_evaluations, label='interpolation')
    ax[0].set_title(f'interpolation with intervals N={num_intervals}')

    # plotting error
    err = abs(y_interpolation_evaluations - f_exact)
    ax[1].semilogy(x_evaluation_pts, err)
    ax[1].set_title('error')

    plt.savefig(f'{file_prefix}spline-interpolation.png')
    return y_interpolation_evaluations
