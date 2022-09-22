import numpy as np
from matplotlib import pyplot as plt

import mypkg


def compute_order(x, xstar):
    # p_{n+1}-p (from the second index to the end)
    diff1 = np.abs(x[1::] - xstar)
    # p_n-p (from the first index to the second to last)
    diff2 = np.abs(x[0:-1] - xstar)
    # linear fit to log of differences
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ' + str(np.exp(fit[1])))
    print('alpha = ' + str(fit[0]))
    return [fit, diff1, diff2]


def aitkens(x, xstar, tol):
    count = len(x)
    hat_x = np.zeros((count - 2, 1))
    for n in range(0, count - 2):
        # compute new iterates using aitkens formula
        hat_x[n] = x[n] - (x[n + 1] - x[n]) ** 2 / (x[n + 2] - 2 * x[n + 1] + x[n])
        # return if we're less than tol
        if np.abs(hat_x[n] - xstar) < tol:
            return hat_x[0:n + 1]
    return hat_x


########################################################
if __name__ == '__main__':

    find = mypkg.Iteration1D(lambda x: np.power(10 / (x + 4), 1.0 / 2.0), 'fixedpt')
    find.p0 = 1.5
    find.tol = 1e-10
    find.Nmax = 100

    find.root()

    xstar = find.pstar
    x = find.vector_of_approximations

    hat_x = aitkens(x, xstar, 1e-10)

    # compute the convergence rate and constant
    print('\nrate and constants for fixedpt:')
    [fit, diff1, diff2] = compute_order(x, xstar)
    print('\nrate and constants for aitkens')
    [fit1, diff11, diff21] = compute_order(hat_x, xstar)
    # plot the data
    plt.loglog(diff2, diff1, 'ro', label='fixedpt data')
    plt.loglog(diff21, diff11, 'gx', label='aitkens data')
    # plot the fits
    plt.loglog(diff2, np.exp(fit[1] + fit[0] * np.log(diff2)), 'r-', label='fixedpt fit')
    plt.loglog(diff21, np.exp(fit1[1] + fit1[0] * np.log(diff21)), 'g-', label='aitkens fit')
    # label the plot axes and create legend
    plt.xlabel('$|p_{n}-p|$')
    plt.ylabel('$|p_{n+1}-p|$')
    plt.legend()
    plt.show()
