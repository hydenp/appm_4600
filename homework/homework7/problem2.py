import numpy as np

from interpolation import barycentric, lagrange, plot_int

if __name__ == '__main__':
    # set configuration
    N = 200
    SAMPLE_POINTS = 1_000
    A = 0
    B = 1


    def f(x):
        return 1 / (1 + (10 * x) ** 2)

    x_int_pts = np.linspace(A, B, N)
    x_eval_pts = np.linspace(A, B, SAMPLE_POINTS)
    f_exact = [f(x) for x in x_eval_pts]

    lagrange_pts = lagrange(f, x_int_pts, x_eval_pts)
    plot_int(x_eval_pts, f_exact, lagrange_pts, N, save=True, file_prefix='prblm2-lagrange')

    barycentric_pts = barycentric(f, x_int_pts, x_eval_pts)
    plot_int(x_eval_pts, f_exact, barycentric_pts, N, save=True, file_prefix='prblm2-barrycentric')
