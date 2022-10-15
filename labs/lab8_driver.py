import numpy as np

from interpolation import natural_spline

if __name__ == '__main__':

    # f = lambda x: math.exp(x)
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1

    # create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a, b, Neval)

    # number of intervals
    Nint = 10

    # evaluate the linear spline
    yeval = natural_spline(f, Nint, a, b, Neval, file_prefix='./plots/lab8/')
