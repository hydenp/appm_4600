from typing import Callable

import numpy as np

from rootfinder.newton_nd_helpers import evaluate_jacobian_inverse, evaluate_f


def broyden_nd(functions: [[Callable]],
               jacobian_functions: [[Callable]], x0: [int], tolerance: float, max_iterations: int):
    """
    :param functions:
    :param jacobian_functions:
    :param x0:
    :param tolerance:
    :param max_iterations:
    :return: root, error_code, iterations
    """

    '''Sherman-Morrison 
   (inv_jacobian+xy^T)^{-1} = inv_jacobian^{-1}-1/p*(inv_jacobian^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''

    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''

    '''initialize with 1 newton step'''

    # jacobian = evalJ(x0)
    jacobian = np.eye(len(jacobian_functions), len(jacobian_functions[0]))
    for row_index, row in enumerate(jacobian_functions):
        for col_index, function in enumerate(row):
            jacobian[row_index][col_index] = function(x0)

    # f_at_xk = evalF(x0)
    f_at_xk = evaluate_f(functions, x0)
    f_at_xk = np.reshape(f_at_xk, (1, len(f_at_xk)))[0]

    # inv_jacobian = np.linalg.inv(jacobian)
    inv_jacobian = evaluate_jacobian_inverse(jacobian_functions, x0)

    s = np.reshape(-inv_jacobian.dot(f_at_xk), (1, len(f_at_xk)))[0]
    xk = x0 + s
    for iteration in range(max_iterations):
        '''(save f_at_xk from previous step)'''
        w = f_at_xk

        '''create new f_at_xk'''
        # f_at_xk = evaluate_f(functions, xk)
        f_at_xk = evaluate_f(functions, np.reshape(xk, (1, len(xk)))[0])
        f_at_xk = np.reshape(f_at_xk, (1, len(f_at_xk)))[0]

        '''y_k = F(xk)-F(xk-1)'''
        y = f_at_xk - w

        '''-A_{k-1}^{-1}y_k'''
        z = -inv_jacobian.dot(y)

        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s, z)
        u = np.dot(s, inv_jacobian)

        ''' inv_jacobian = A_k^{-1} via Morrison formula'''
        tmp = s + z
        tmp2 = np.outer(tmp, u)
        inv_jacobian = inv_jacobian + 1.0 / p * tmp2

        ''' -A_k^{-1}F(x_k)'''
        s = -inv_jacobian.dot(f_at_xk)
        xk = xk + s
        if np.linalg.norm(s) < tolerance:
            # alpha = xk
            # ier = 0
            return xk, 0, iteration + 1

    # alpha = xk
    # ier = 1
    return xk, 1, max_iterations
