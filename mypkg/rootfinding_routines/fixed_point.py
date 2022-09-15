import numpy as np


def fixedpt_with_approximations(f, x0, tol, Nmax):
    """x0 = initial guess"""
    """Nmax = max number of iterations"""
    """tol = stopping tolerance"""

    approximations = np.zeros((Nmax, 1))

    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        approximations[count] = x1
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            approximations = approximations[:count]
            return [xstar, approximations, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, approximations, ier]