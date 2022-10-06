import math

import numpy
from matplotlib import pyplot as plt


# prelab lab 6
def forward_difference(f, s, h):
    return (f(s + h) - f(s)) / h


def centered_difference(f, s, h):
    return (f(s + h) - f(s - h)) / (2 * h)


H = 0.01 * 2.0 ** (-numpy.arange(0, 10))
S = math.pi/2


if __name__ == '__main__':

    forward_difference_approximates = []
    centered_difference_approximates = []
    for h_i in H:
        forward_difference_approximates.append(
            forward_difference(lambda x: math.cos(x), h_i, S)
        )
        centered_difference_approximates.append(
            centered_difference(lambda x: math.cos(x), h_i, S)
        )

    print(f'forward difference root: {forward_difference_approximates[-1]}')
    print(f'centered difference root: {centered_difference_approximates[-1]}')

    ACTUAL_ROOT = -1

    centered_abs_errors = [abs(ACTUAL_ROOT) - abs(x) for x in centered_difference_approximates]
    forward_abs_errors = [abs(ACTUAL_ROOT) - abs(x) for x in forward_difference_approximates]

    plt.plot(H, centered_abs_errors, label='centered difference')
    plt.plot(H, forward_abs_errors, label='forward difference')
    plt.legend(loc='lower right')
    plt.title('Absolute error differences')
    plt.show()

