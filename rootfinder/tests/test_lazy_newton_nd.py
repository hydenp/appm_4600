import math
from unittest import TestCase

from rootfinder.lazy_newton_nd import lazy_newton_nd, evaluate_jacobian_inverse, evaluate_f


class TestLazyNewtonND(TestCase):
    def setUp(self) -> None:
        """
        Let F be defined as F([g(x_1, x_2), h(x_1, x_2)])
        where
        g(x1, x2) = 3*x_1^2 - x_2^2
        h(x1, x2) = 3*x_1*x_2^2 - x_1^3 - 1

        it is known there is a root at (x_1, x_2) = (1/2, sqrt(3)/2)
        """
        self.F = [
            [lambda x: 3*x[0]**2 - x[1]**2],
            [lambda x: 3*x[0]*x[1]**2 - x[0]**3 - 1]
        ]

        """
        Jacobian functions defines as
        [
            [ g_x, g_y]
            [ h_x, h__y]
        ]
        """

        self.jacobian_funcs = [
            [lambda x: 6*x[0], lambda x: -2*x[1]],
            [lambda x: 3*x[1]**2 - 3*x[0]**2, lambda x: 6*x[0]*x[1]]
        ]

    def test_evaluate_f_at_x(self):
        x = [1, 1]
        res = evaluate_f(self.F, x)
        self.assertEqual(2, res[0][0])
        self.assertEqual(1, res[1][0])

        x = [0, 0]
        res = evaluate_f(self.F, x)
        self.assertEqual(0, res[0][0])
        self.assertEqual(-1, res[1][0])

    def test_evaluate_jacobian_inverse(self):
        x = [1, 1]
        res = evaluate_jacobian_inverse(self.jacobian_funcs, x)

        self.assertAlmostEqual(1/6, res[0][0])
        self.assertAlmostEqual(1/18, res[0][1])
        self.assertAlmostEqual(0, res[1][0])
        self.assertAlmostEqual(1/6, res[1][1])

    def test_lazy_newton(self):

        roots, err_code, iterations = lazy_newton_nd(self.F, self.jacobian_funcs, [1, 1], 10e-10, 100)
        self.assertAlmostEqual(1/2, roots[0][0])
        self.assertAlmostEqual(math.sqrt(3)/2, roots[1][0])





