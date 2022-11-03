# get lgwts routine and numpy
from typing import Callable

from gauss_legendre import *


# Adaptive quadrature subroutines
#   - One can replace calls to eval_gauss_quad() in
#   - adaptive_quad() with different quadrature rules
#   - on [a,b]

def eval_gauss_quad(M, a, b, f):
    """
    Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
    Input:
      M - number of quadrature nodes per
      a,b - interval [a,b]
      f - function to integrate

    Output:
      I_hat - approx integral
      x - quadrature nodes
      w - quadrature weights

    Currently uses Gauss-Legendre rule
    """
    x, w = lgwt(M, a, b)
    I_hat = np.sum(f(x) * w)
    return I_hat, x, w


def composite_simpson(num_intervals: int, a: float, b: float, f: Callable):
    h = (b - a) / num_intervals

    xi_0 = f(a) + f(b)
    xi_1 = 0
    xi_2 = 0

    for i in range(1, num_intervals):
        x = a + i * h

        if i % 2 == 0:
            xi_2 += f(x)
        else:
            xi_1 += f(x)

    x = [a + i * h for i in range(num_intervals + 1)]

    return h * (xi_0 + 2 * xi_2 + 4 * xi_1) / 3, x, None


def trapezoidal(num_intervals: int, a: float, b: float, f: Callable):
    intervals = np.linspace(a, b, num_intervals + 1)
    delta_x = (b - a) / num_intervals

    sum_f = 0
    for x_i in intervals:
        if x_i in [intervals[0], intervals[-1]]:
            sum_f += f(x_i)
        else:
            sum_f += 2 * f(x_i)

    return delta_x / 2 * sum_f, intervals, None


def adaptive_quad(a, b, f, tol, M, method):
    """
    Adaptive numerical integrator for \int_a^b f(x)dx

    Input:
    a,b - interval [a,b]
    f - function to integrate
    tol - absolute accuracy goal
    M - number of quadrature nodes per bisected interval

    Output: I - the approximate integral
            X - final adapted grid nodes
    """
    # 1/2^50 ~ 1e-15
    maxit = 50
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit, 1))
    left_p[0] = a
    right_p[0] = b
    # initial approx and grid
    s[0], x, _ = method(M, a, b, f)
    # save grid
    X = []
    X.append(x)
    j = 1
    I = 0
    while j < maxit:
        # get midpoint to split interval into left and right
        c = 0.5 * (left_p[j - 1] + right_p[j - 1])
        # compute integral on left and right spilt intervals
        s1, x, _ = method(M, left_p[j - 1], c, f)
        X.append(x)
        s2, x, _ = method(M, c, right_p[j - 1], f)
        X.append(x)
        if np.max(np.abs(s1 + s2 - s[j - 1])) > tol:
            left_p[j] = left_p[j - 1]
            right_p[j] = 0.5 * (left_p[j - 1] + right_p[j - 1])
            s[j] = s1
            left_p[j - 1] = 0.5 * (left_p[j - 1] + right_p[j - 1])
            s[j - 1] = s2
            j = j + 1
        else:
            I = I + s1 + s2
            j = j - 1
            if j == 0:
                j = maxit
    return I, np.unique(X)
