import numpy as np
from matplotlib import pyplot as plt

from interpolation import eval_lin_spline

if __name__ == '__main__':

    # f = lambda x: math.exp(x)
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval = np.linspace(a, b, Neval)

    ''' number of intervals'''
    Nint = 10

    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)

    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
        fex[j] = f(xeval[j])

    plt.plot(xeval, fex, label='exact')
    plt.plot(xeval, yeval, label='interpolation')
    plt.title(f'interpolation with {Nint}')
    plt.savefig('./plots/lab8/spline-interpolation.png')
    plt.cla()

    err = abs(yeval - fex)
    plt.plot(xeval, err)
    plt.title('error')
    plt.savefig('./plots/lab8/spline-interpolation-error.png')
    plt.cla()
