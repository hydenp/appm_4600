import numpy as np

from interpolation import newton_dd
from interpolation.monomial_basis import monomial_basis

if __name__ == '__main__':

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    def f(x): return 1.0 / (1 + np.power(10.0 * x, 2))

    SAMPLE_SIZE = 1_000

    for d in degrees:

        _ = monomial_basis(f, d, SAMPLE_SIZE, 'save')

        newton_dd(f, d, 0, 1, SAMPLE_SIZE, 'save')
