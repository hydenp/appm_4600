import time

import numpy as np
import numpy.linalg as la
import scipy.linalg as scila


def driver():
    ''' create  matrix for testing different ways of solving a square
    linear system'''

    '''' N = size of system'''
    N = 100

    ''' Right hand side'''
    b = np.random.rand(N, 1)
    A = np.random.rand(N, N)

    x = scila.solve(A, b)

    test = np.matmul(A, x)
    r = la.norm(test - b)

    print(r)

    ''' Create an ill-conditioned rectangular matrix '''

    ns = [100, 500, 1000, 2000, 4000, 5000]
    for N in ns:
        # N = 10
        # M = 5
        A = create_rect(N, N)
        b = np.random.rand(N, 1)

        # solve the
        start = time.process_time()
        lu, piv = scila.lu_factor(A)
        duration = time.process_time() - start
        print(f'Time to LU factor with N={N} : {duration}')

        start = time.process_time()
        scila.lu_solve((lu, piv), b)
        duration = time.process_time() - start
        print(f'Time to solve LU with N={N} : {duration}')

        # ---
        start = time.process_time()
        scila.qr(A)
        duration = time.process_time() - start
        print(f'Time to factor with QR N={N} : {duration}')
        print('---')
        print()

def create_rect(N, M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1, 10, M)
    d = 10 ** (-a)

    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N, N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1, R)
    A = np.random.rand(M, M)
    Q2, R = la.qr(A)
    test = np.matmul(Q2, R)

    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
