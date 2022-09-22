import numpy as np


class Iteration1D:
    def __init__(self, f, method, f_prime=None, f_double_prime=None):
        # we assign to or initialize as None all self attributes
        self.f = f
        self.f_prime = f_prime
        self.f_double_prime = f_double_prime

        self.method = method
        # initial interval for bisection:
        self.a = None
        self.b = None
        # initial guess for newton/fixed point
        self.p0 = None
        # tolerance and max iter
        self.tol = None
        self.Nmax = None
        # info message
        self.info = None
        # root
        self.pstar = None
        # iters for newton or fixed point
        self.p_iters = None
        # the vector of approximations made in fixed point iteration
        self.vector_of_approximations = None

    def root(self) -> None:
        if self.method == 'bisection':
            # add checks and bisection routine
            if self.f is None or self.a is None or self.b is None or self.tol is None or self.Nmax is None:
                raise Exception('error: some attributes for bisection aresys. returning ..')

            self.pstar, self.info = bisection(self.f, self.a, self.b, self.tol, self.Nmax)
        elif self.method == 'fixedpt':
            # add checks and fixed point routine
            if self.f is None or self.p0 is None or self.tol is None or self.Nmax is None:
                raise Exception('error: some attributes for fixedpt are unset. returning ..')

            self.pstar, self.vector_of_approximations, self.info = fixed_point(self.f, self.p0, self.tol,
                                                                               self.Nmax)
        elif self.method == 'newton':
            self.vector_of_approximations, self.pstar, self.info, self.p_iters = newton(self.f, self.f_prime,
                                                                                        self.p0, self.tol,
                                                                                        self.Nmax)

        elif self.method == 'newton-improved':
            new_guess, err_code = bisection_to_neighborhood(self.f, self.f_prime, self.f_double_prime, self.a,
                                                            self.b, self.Nmax)

            self.vector_of_approximations, self.pstar, self.info, self.p_iters = newton(self.f, self.f_prime,
                                                                                        new_guess, self.tol,
                                                                                        self.Nmax)


def bisection_to_neighborhood(f, f_prime, f_double_prime, a, b, Nmax):
    """
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    """

    # """first verify there is a root we can find in the interval"""
    # if fa * fb > 0:
    #     ier = 1
    #     astar = a
    #     return [astar, ier]

    def g_prime(x):
        return f(x) * f_double_prime(x) / f_prime(x) ** 2

    """verify end point is not a root"""
    if abs(g_prime(a)) == 0:
        return [a, 0]

    if abs(g_prime(b)) == 0:
        return [b, 0]

    count = 0
    while count < Nmax:
        c = 0.5 * (a + b)

        if abs(g_prime(c)) < 1:
            """
            return c which will be the initial guess
            """
            return [c, 0]

        # choose which interval to continue with
        # it will still fail with even multiplicity roots like bisection
        fa = f(a)
        fc = f(c)
        if fa * fc < 0:
            b = c
        else:
            a = c

        count += 1

    raise Exception("No valid neighborhood found")


def bisection(f, a, b, tol, Nmax):
    """
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
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
        ier = 1
        astar = a
        return [astar, ier]

    """verify end point is not a root"""
    if fa == 0:
        astar = a
        ier = 0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    while count < Nmax:
        c = 0.5 * (a + b)
        fc = f(c)

        if fc == 0:
            astar = c
            ier = 0
            return [astar, ier]

        if fa * fc < 0:
            b = c
        elif fb * fc < 0:
            a = c
            fa = fc
        else:
            astar = c
            ier = 3
            return [astar, ier]

        if abs(b - a) < tol:
            astar = a
            ier = 0
            return [astar, ier]

        count = count + 1

    astar = a
    ier = 2
    return [astar, ier]


def fixed_point(f, x0, tol, Nmax):
    """x0 = initial guess"""
    """Nmax = max number of iterations"""
    """tol = stopping tolerance"""

    approximations = np.zeros((Nmax + 1, 1))

    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        approximations[count] = x1
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            approximations = approximations[:count]
            return [xstar, approximations, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, approximations, ier]


def newton(f, fp, p0, tol, Nmax):
    """
    Newton iteration.

    Inputs:
      f,fp - function and derivative
      p0   - initial guess for root
      tol  - iteration stops when p_n,p_{n+1} are within tol
      Nmax - max number of iterations
    Returns:
      p     - an array of the iterates
      pstar - the last iterate
      info  - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)
    """
    p = [0 for _ in range(Nmax+1)]
    p[0] = p0
    for it in range(Nmax):
        p1 = p0 - f(p0) / fp(p0)
        p[it + 1] = p1
        if abs(p1 - p0) < tol:
            pstar = p1
            info = 0
            return [p, pstar, info, it]
        p0 = p1
    pstar = p1
    info = 1
    return [p, pstar, info, it]
