# page 206 in the text book
from typing import Callable


def composite_simpson(f: Callable, a: float, b: float, num_intervals: int):
    h = (b - a) / num_intervals

    xi_0 = f(a) + f(b)
    xi_1 = 0
    xi_2 = 0

    for i in range(1, num_intervals):
        x = a + i * h

        if i % 2 == 0:
            xi_2 += f(x)
        else:
            xi_1 += f(x)

    return h * (xi_0 + 2 * xi_2 + 4 * xi_1) / 3


# if __name__ == '__main__':
#     def fun(x): return x ** 5
#
#     print(composite_simpson(fun, 0, 5, 20))
