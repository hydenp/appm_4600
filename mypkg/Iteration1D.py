import numpy as np
from matplotlib import pyplot as plt

from rootfinding_routines import bisection, fixed_point


class Iteration1D:
    def __init__(self, f, method):
        # we assign to or initialize as None all self attributes
        self.f = f
        self.method = method
        # initial interval for bisection:
        self.a = None
        self.b = None
        # initial guess for newton/fixedpt
        self.p0 = None
        # tolerance and max iter
        self.tol = None
        self.Nmax = None
        # info message
        self.info = None
        # root
        self.pstar = None
        # iters for newton or fixedpt
        self.p_iters = None
        # the vector of approximations made in fixed point iteration
        self.fixed_pt_approximations = None
        self.fit = None
        self.diff1 = None
        self.diff2 = None

    def root(self) -> None:
        if self.method == 'bisection':
            # add checks and bisection routine
            if self.f is None or self.a is None or self.b is None or self.tol is None or self.Nmax is None:
                print('error: some attributes for bisection aresys. returning ..')
                return

            self.pstar, self.info = bisection.bisection(self.f, self.a, self.b, self.tol, self.Nmax)
        elif self.method == 'fixedpt':
            # add checks and fixed point routine
            if self.f is None or self.p0 is None or self.tol is None or self.Nmax is None:
                print('error: some attributes for fixedpt are unset. returning ..')
                return

            self.pstar, self.fixed_pt_approximations, self.info = \
                fixed_point.fixedpt_with_approximations(self.f,
                                                        self.p0,
                                                        self.tol,
                                                        self.Nmax)

    def compute_order(self) -> None:
        """
        Note that a fixed point iterations must be run before computing the order
        """

        if self.fixed_pt_approximations is None:
            raise ValueError("fixed_pt_approximations - must run fixed point iteration before computing order")

        # p_{n+1}-p (from the second index to the end)
        self.diff1 = np.abs(self.fixed_pt_approximations[1::] - self.pstar)
        # p_n-p (from the first index to the second to last)
        self.diff2 = np.abs(self.fixed_pt_approximations[0:-1] - self.pstar)

        # linear fit to log of differences
        self.fit = np.polyfit(np.log(self.diff2.flatten()), np.log(self.diff1.flatten()), 1)

        print('the order equation is')
        print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
        print('lambda = ' + str(np.exp(self.fit[1])))
        print('alpha = ' + str(self.fit[0]))

    def plot_fit(self) -> None:
        """
        Fixed point and order computation before plotting the fit
        """
        if self.fit is None:
            raise ValueError("calculated fit is None - must run order calculation plotting fit")

        plt.loglog(self.diff2, self.diff1, 'ro', label='data')
        # plot the fit
        plt.loglog(self.diff2, np.exp(self.fit[1] + self.fit[0] * np.log(self.diff2)), 'b-', label='fit')
        plt.xlabel('$|p_{n}-p|$')
        plt.ylabel('$|p_{n+1}-p|$')
        plt.legend()
        plt.show()

# def bisection(f, a, b, tol, Nmax):
#     """
#     Inputs:
#       f,a,b       - function and endpoints of initial interval
#       tol, Nmax   - bisection stops when interval length < tol
#                   - or if Nmax iterations have occured
#     Returns:
#       astar - approximation of root
#       ier   - error message
#             - ier = 1 => cannot tell if there is a root in the interval
#             - ier = 0 == success
#             - ier = 2 => ran out of iterations
#             - ier = 3 => other error ==== You can explain
#     """
#
#     """first verify there is a root we can find in the interval"""
#     fa = f(a)
#     fb = f(b)
#     if fa * fb > 0:
#         ier = 1
#         astar = a
#         return [astar, ier]
#
#     """verify end point is not a root"""
#     if fa == 0:
#         astar = a
#         ier = 0
#         return [astar, ier]
#
#     if fb == 0:
#         astar = b
#         ier = 0
#         return [astar, ier]
#
#     count = 0
#     while count < Nmax:
#         c = 0.5 * (a + b)
#         fc = f(c)
#
#         if fc == 0:
#             astar = c
#             ier = 0
#             return [astar, ier]
#
#         if fa * fc < 0:
#             b = c
#         elif fb * fc < 0:
#             a = c
#             fa = fc
#         else:
#             astar = c
#             ier = 3
#             return [astar, ier]
#
#         if abs(b - a) < tol:
#             astar = a
#             ier = 0
#             return [astar, ier]
#
#         count = count + 1
#
#     astar = a
#     ier = 2
#     return [astar, ier]


# def fixedpt(f, x0, tol, Nmax):
#     """x0 = initial guess"""
#     """Nmax = max number of iterations"""
#     """tol = stopping tolerance"""
#
#     count = 0
#     while count < Nmax:
#         count = count + 1
#         x1 = f(x0)
#         if abs(x1 - x0) < tol:
#             xstar = x1
#             ier = 0
#             return [xstar, ier]
#         x0 = x1
#
#     xstar = x1
#     ier = 1
#     return [xstar, ier]
#
#
# def fixedpt_with_approximations(f, x0, tol, Nmax):
#     """x0 = initial guess"""
#     """Nmax = max number of iterations"""
#     """tol = stopping tolerance"""
#
#     approximations = np.zeros((Nmax, 1))
#
#     count = 0
#     while count < Nmax:
#         count = count + 1
#         x1 = f(x0)
#         approximations[count] = x1
#         if abs(x1 - x0) < tol:
#             xstar = x1
#             ier = 0
#             approximations = approximations[:count]
#             return [xstar, approximations, ier]
#         x0 = x1
#
#     xstar = x1
#     ier = 1
#     return [xstar, approximations, ier]
