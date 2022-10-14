import math

from interpolation import barycentric, lagrange

if __name__ == '__main__':
    # set configuration
    NUM_INTERPOLATION_NODES = 200
    NUM_SAMPLE_POINTS = 2_000
    A = -1
    B = 1

    # construct Chebychev's points
    chebychev_pts = []
    for i in range(2 * NUM_INTERPOLATION_NODES + 1):
        chebychev_pts.append(
            math.cos(((2 * i - 1) * math.pi) / (2 * NUM_INTERPOLATION_NODES))
        )

    # define function
    def f(x):
        return 1 / (1 + (10 * x) ** 2)


    lagrange(f, NUM_INTERPOLATION_NODES, A, B, NUM_SAMPLE_POINTS, special_x=chebychev_pts, file_prefix='prblm3-')
    barycentric(f, NUM_INTERPOLATION_NODES, A, B, NUM_SAMPLE_POINTS, special_x=chebychev_pts, file_prefix='prblm3-')
