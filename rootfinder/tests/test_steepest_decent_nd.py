import math
from unittest import TestCase


class Test(TestCase):
    def setUp(self):
        self.F_FUNCS = [
            [lambda x: x[0] + math.cos(x[0] * x[1] * x[2]) - 1],
            [lambda x: pow((1 - x[0]), 0.25) + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1],
            [lambda x: -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1]
        ]

        self.JACOBIAN_FUNCTIONS = [
            [lambda x: 1 - x[1] * x[2] * math.sin(x[0] * x[1] * x[2]), lambda x: None, lambda x: None],
            [lambda x: -1 * 0.25 * pow((1 - x[0]), -0.75), lambda x: 1, lambda x: 2 * 0.05 * x[2] - 0.15],
            [lambda x: -2 * x[0], lambda x: -0.1 * 2 * x[1], lambda x: 1],
        ]

        self.tolerance = 10e-8
        self.max_iterations = 100

    # def test_evaluate_gradient(self):
    #     x = [0, 0, 1]
    #     res = evaluate_gradient(self.GRADIENT_FUNCS, x)
    #     print(res)
    #     self.assertAlmostEqual(1, res[0])
    #     self.assertAlmostEqual(1, res[1])

    def test_steepest_descent_nd(self):
        pass
        # x_0 = [0, -2, -2]
        # # x_0 = [0, 0, 0]
        # x_0 = [1, 1, 1]
        # # x_0 = [0, 0.15, 1]
        #
        # self.max_iterations = 100
        #
        # t = evaluate_f(self.F_FUNCS, [0, -2, -2])
        # # print(t)
        #
        # res = steepest_descent_nd(self.F_FUNCS, self.JACOBIAN_FUNCTIONS, x_0, self.tolerance, self.max_iterations)
        # print(res)
        #
        # print(evaluate_f(self.F_FUNCS, res[0]))

        # t = evaluate_f(self.F_FUNCS, [-3.81368159e-17, 1.00000000e-01, 1.00000000e+00])
        # print(t)
