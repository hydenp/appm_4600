import numpy as np

import mypkg

if __name__ == '__main__':
    # instantiate Iteration1D object using bisection method
    find = mypkg.Iteration1D(lambda x: (x ** 2) * (x - 1), 'bisection')

    # set tol and Nmax
    find.tol = 1e-6
    find.Nmax = 100

    ########################################################
    # Using bisection

    # PART A
    find.a = 0.5
    find.b = 2.0
    # find the root
    find.root()
    print(f'Part A: bisection root {find.pstar}')

    # PART B
    find.a = -1.0
    find.b = 0.5
    # set tol and Nmax
    # find the root
    find.root()
    print(f'Part B: bisection root {find.pstar}')

    # PART C
    find.a = -1.0
    find.b = 2.0
    # set tol and Nmax
    # find the root
    find.root()
    print(f'Part C: bisection root {find.pstar}')

    ########################################################
    # Using fixed point
    find.method = 'fixedpt'

    # PART A
    find.f = lambda x: np.power(x, 2/3)

    find.p0 = 1.2
    find.root()
    print(f'When f(x)=x^3/2 guess 1.2: fixed point root {find.pstar}')

    find.p0 = 0.1
    find.root()
    print(f'When f(x)=x^3/2 guess 0.1: fixed point root {find.pstar}')

    # PART B
    find.f = lambda x: np.power(x, 1.5)

    find.p0 = 1.2
    find.root()
    print(f'When f(x)=x^2/3 guess 1.2: fixed point root {find.pstar}')

    find.p0 = 0.1
    find.root()
    print(f'When f(x)=x^2/3 guess 0.1: fixed point root {find.pstar}')
