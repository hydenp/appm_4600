import csv

import numpy as np

from interpolation import create_vandermonde, monomial_basis

if __name__ == '__main__':
    # set configuration
    NUM_INTERPOLATION_NODES = 20
    SAMPLE_POINTS = 100


    def f(x_i):
        return 1 / (1 + (10 * x_i) ** 2)


    # create the vandermonde
    max_polynomial_degree = 10
    vandermonde_matrix, _ = create_vandermonde(max_polynomial_degree)
    with open('vandermonde.csv', "w", newline="") as s:
        writer = csv.writer(s)
        writer.writerows(vandermonde_matrix.round(5))

    # running an interpolation
    coefficients = monomial_basis(f, NUM_INTERPOLATION_NODES, SAMPLE_POINTS, plot_save_show='save',
                                  file_prefix='prblm1-')

    # storing coefficients
    with open('coefficients.csv', "w", newline="") as s:
        writer = csv.writer(s)
        r = np.reshape(coefficients, (len(coefficients), 1))
        writer.writerows(r)
