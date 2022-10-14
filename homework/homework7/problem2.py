from interpolation import barycentric, lagrange

if __name__ == '__main__':
    # set configuration
    NUM_INTERPOLATION_POINTS = 200
    SAMPLE_POINTS = 2_000
    A = 0
    B = 1


    def f(x):
        return 1 / (1 + (10 * x) ** 2)


    lagrange(f, NUM_INTERPOLATION_POINTS, A, B, SAMPLE_POINTS, file_prefix='prblm2-')
    barycentric(f, NUM_INTERPOLATION_POINTS, A, B, SAMPLE_POINTS, file_prefix='prblm2-')
