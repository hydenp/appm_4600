import numpy as np

from interpolation import hermite

if __name__ == '__main__':
    # function to interpolate
    def f(x):
        return 1 / (1 + x ** 2)

    def fp(x):
        return -2*x / (1 + x ** 2) ** 2

    N = 5
    x_int_nodes = np.linspace(-5, 5, N)
    x_eval_pts = np.linspace(-5, 5, 50)

    # how many points we want to evaluate at after creating our interpolation
    NUM_SAMPLE_POINTS = 10
    # number of intervals for the spline evaluations
    NUM_INTERVALS = 20

    N = 5
    # lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')

    hermite(f, fp, x_int_nodes, x_eval_pts)


    # NUM_INTERPOLATION_NODES = 10
    # lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    #
    # NUM_INTERPOLATION_NODES = 15
    # lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    #
    # NUM_INTERPOLATION_NODES = 20
    # lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
