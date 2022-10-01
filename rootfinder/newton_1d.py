from typing import Callable


def newton_1d(f: Callable, f_prime: Callable, p0: float, tolerance: float, max_iterations: int):
    """
    Newton iteration.

    Inputs:
      f,fp - function and derivative
      p0   - initial guess for root
      tol  - iteration stops when p_n,p_{n+1} are within tol
      max_iterations - max number of iterations
    Returns:
      p     - an array of the iterates
      pstar - the last iterate
      ier  - success message
            - 0 if we met tol
            - 1 if we hit max_iterations (fail)
    """
    p1 = None
    p = [p0]

    assert max_iterations > 0

    for iteration in range(max_iterations):
        p1 = p0 - f(p0) / f_prime(p0)
        p.append(p1)
        if abs(p1 - p0) < tolerance:
            # pstar = p1
            # ier = 0
            return p, p1, 0, iteration
        p0 = p1

    # pstar = p1
    # ier = 1
    return p, p1, 1, max_iterations
