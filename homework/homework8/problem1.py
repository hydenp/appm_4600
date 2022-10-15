from interpolation import lagrange, natural_spline

if __name__ == '__main__':
    # function to interpolate
    def f(x):
        return 1 / (1 + x ** 2)

    # how many points we want to evaluate at after creating our interpolation
    NUM_SAMPLE_POINTS = 100
    # number of intervals for the spline evaluations
    NUM_INTERVALS = 20

    NUM_INTERPOLATION_NODES = 5
    lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    natural_spline(f, NUM_INTERVALS, -5, 5, NUM_SAMPLE_POINTS,
                   file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')

    NUM_INTERPOLATION_NODES = 10
    lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    natural_spline(f, NUM_INTERVALS, -5, 5, NUM_SAMPLE_POINTS,
                   file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')

    NUM_INTERPOLATION_NODES = 15
    lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    natural_spline(f, NUM_INTERVALS, -5, 5, NUM_SAMPLE_POINTS,
                   file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')

    NUM_INTERPOLATION_NODES = 20
    lagrange(f, NUM_INTERPOLATION_NODES, -5, 5, NUM_SAMPLE_POINTS, file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
    natural_spline(f, NUM_INTERVALS, -5, 5, NUM_SAMPLE_POINTS,
                   file_prefix=f'prblm1-n-{NUM_INTERPOLATION_NODES}_')
