import numpy as np

import mypkg

if __name__ == '__main__':
    # instantiate Iteration1D object using bisection method
    find = mypkg.Iteration1D(lambda x: np.power(10 / (x + 4), 1 / 2), 'fixedpt')
    find.p0 = 1.5
    find.tol = 1e-10
    find.Nmax = 100

    find.root()
    find.compute_order()
    find.plot_fit()
