import numpy as np


def get_sub_interval_indexes(a: float, b: float, points: []) -> []:
    above = np.where(points >= a)[0]
    below = np.where(points <= b)[0]
    return list(set(above) & set(below))


def line_through(x_0: float, x_1: float, f_0: float, f_1: float) -> [(float, float)]:
    return lambda x: (f_1 - f_0) / (x_1 - x_0) * (x - x_0) + f_0


def eval_lin_spline(xeval, Neval, a, b, f, Nint):
    """create the intervals for piecewise approximations"""
    xint = np.linspace(a, b, Nint + 1)

    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        ind = get_sub_interval_indexes(xint[jint], xint[jint + 1], xeval)
        n = len(ind)

        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)

        line = line_through(a1, b1, fa1, fb1)
        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points 
            in the interval'''
            '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
            the points (a1,fa1) and (b1,fb1)'''

            yeval[ind[kk]] = line(xeval[ind[kk]])

    return yeval
