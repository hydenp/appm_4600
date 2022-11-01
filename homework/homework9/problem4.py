import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 5, 100)


# exact plot
def f(x):
    return (
        x - (x**3/6) + (x**5/120) - (x**7/5040) + (x**9/362_880) - (x**11/39_916_800)
    )


x_exact = np.array([f(x_i) for x_i in x])


# part A
def f_a(x):
    return (x - 7 * x ** 3 / 60) / (1 + x ** 2 / 20)


x_a = np.array([f_a(x_i) for x_i in x])


# part B
def f_b(x):
    return x / (1 + x ** 2 / 6 + 7 * x ** 4 / 360)


x_b = np.array([f_b(x_i) for x_i in x])


# part C
def f_c(x):
    return (x - 7 * x ** 3 / 60) / (1 + x ** 2 / 20)


x_c = np.array([f_c(x_i) for x_i in x])


a_error = abs(x_exact - x_a)
b_error = abs(x_exact - x_b)
c_error = abs(x_exact - x_c)

if __name__ == '__main__':
    plt.semilogy(x, a_error, label='part a & c error')
    plt.semilogy(x, b_error, label='part b error')
    plt.semilogy(x, c_error, label='part c error')
    plt.title("Error of Plate vs McLauren")
    plt.legend(loc='upper left')
    plt.savefig('err-problem-4.png')
