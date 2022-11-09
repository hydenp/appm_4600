from scipy.integrate import quad

from quadrature import composite_simpson, trapezoidal


def f(x):
    return 1 / (1 + x ** 2)


if __name__ == '__main__':
    A = -5
    B = 5
    N = 20
    EXACT_VALUE = 2.746801533890032

    # part A
    trap_result = trapezoidal(f, A, B, N)
    simp_result = composite_simpson(f, A, B, N)
    print('Results with N = 20')
    print('trapezoidal =', trap_result)
    print('trapezoidal error =', abs(trap_result - EXACT_VALUE))
    print('simpson =', simp_result)
    print('simpson error =', abs(simp_result - EXACT_VALUE))
    # results:
    # trapezoidal = 2.7462081624602193
    # simpson = 2.7429080175186034

    # part C
    EPSILON = 10e-6

    prev_result = float('inf')
    for i in range(5, 1_000):
        trap_result = trapezoidal(f, A, B, i)
        if abs(prev_result - trap_result) < EPSILON:
            print(f'Trapezoid requires {i} intervals - for e = {EPSILON}')
            break
        prev_result = trap_result

    prev_result = float('inf')
    for i in range(5, 20_000):
        simp_result = composite_simpson(f, A, B, i)

        if abs(prev_result - simp_result) < EPSILON:
            print(f'Simpson requires {i} intervals - for e = {EPSILON}')
            break
        prev_result = simp_result

    print()

    result, _, info_dict = quad(f, A, B, full_output=1)
    print('neval for scipy quad with default tolerance:', info_dict['neval'])
    print('difference to trapezoid result =', abs(result - trap_result))
    print('difference to simpson result =', abs(result - simp_result))

    result, _, info_dict = quad(f, A, B, epsabs=EPSILON, full_output=1)
    print('neval for scipy quad with error of 10e-4:', info_dict['neval'])
    print('difference to trapezoid result =', abs(result - trap_result))
    print('difference to simpson result =', abs(result - simp_result))
