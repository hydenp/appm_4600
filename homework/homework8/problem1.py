import numpy as np

from interpolation import hermite, lagrange, plot_int

if __name__ == '__main__':
    # function to interpolate
    def f(x):
        return 1 / (1 + x ** 2)

    def fp(x):
        return -2*x / (1 + x ** 2) ** 2

    x_eval_pts = np.linspace(-5, 5, 50)
    f_exact = [f(x) for x in x_eval_pts]

    # number of intervals for the spline evaluations
    NUM_INTERVALS = 20

    Ns = [5, 10, 15, 20]

    for N in Ns:

        x_int_nodes = np.linspace(-5, 5, N)

        lagrange_eval_pts = lagrange(f, x_int_nodes, x_eval_pts)
        plot_int(x_eval_pts, f_exact, lagrange_eval_pts, N, save=True, file_prefix='lagrange')

        hermite_eval_pts = hermite(f, fp, x_int_nodes, x_eval_pts)
        plot_int(x_eval_pts, f_exact, hermite_eval_pts, N, save=True, file_prefix='hermite')
