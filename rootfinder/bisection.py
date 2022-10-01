from typing import Callable


def bisection(f: Callable, a: float, b: float, tolerance: float, max_iterations: int):
    """
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, max_iterations   - bisection stops when interval length < tol
                  - or if max_iterations iterations have max_iterations
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    """

    """first verify there is a root we can find in the interval"""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        # ier = 1
        # astar = a
        return a, 1

    """verify end point is not a root"""
    if fa == 0:
        # astar = a
        # ier = 0
        return a, 0

    if fb == 0:
        # astar = b
        # ier = 0
        return b, 0

    iteration = 0
    while iteration < max_iterations:
        c = 0.5 * (a + b)
        fc = f(c)

        if fc == 0:
            # astar = c
            # ier = 0
            return c, 0

        if fa * fc < 0:
            b = c
        elif fb * fc < 0:
            a = c
            fa = fc
        else:
            # astar = c
            # ier = 3
            return c, 3

        if abs(b - a) < tolerance:
            # astar = a
            # ier = 0
            return a, 0

        iteration += 1

    # astar = a
    # ier = 2
    return a, 2
