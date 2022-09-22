import numpy as np

import mypkg

if __name__ == '__main__':
    def f(x): return np.exp((x ** 2) + 7 * x - 30) - 1


    def f_prime(x): return (np.exp(x ** 2 + 7 * x - 30) - 1) * (2 * x + 7)


    def f_double_prime(x): return \
        (np.exp(x ** 2 + 7 * x - 30) - 1) * (4 * x ** 2 + 28 * x + 51) - 2


    find = mypkg.Iteration1D(f, 'newton', f_prime, f_double_prime)
    find.Nmax = 500
    find.tol = 10e-5

    find.a = 2
    find.b = 4.5
    find.p0 = 4.5
    find.root()
    print(find.pstar)

    find.method = 'newton-improved'
    find.root()
    print(find.pstar)
