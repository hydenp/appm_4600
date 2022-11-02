from scipy.integrate import quad

from quadrature import composite_simpson, trapezoidal


def f(x):
    return 1 / (1 + x ** 2)


if __name__ == '__main__':
    A = -5
    B = 5
    N = 20

    # part A
    print('trapezoidal =', trapezoidal(f, A, B, N))
    print('simpson =', composite_simpson(f, A, B, N))
    # results:
    # trapezoidal = 2.7462081624602193
    # simpson = 2.7429080175186034

    print()

    # part C
    EPSILON = 10e-4

    _, _, info_dict = quad(f, A, B, full_output=1)
    print('neval for scipy quad with default tolerance:', info_dict['neval'])
    _, _, info_dict = quad(f, A, B, epsabs=EPSILON, full_output=1)
    print('neval for scipy quad with error of 10e-4:', info_dict['neval'])
