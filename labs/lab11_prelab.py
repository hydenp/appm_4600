from typing import Callable


# Simpson's Rule
def simpson(f: Callable, a: float, b: float):
    return (
            (b - a) / 6 * (f(a) + 4 * f((b - a) / 2) + f(b))
    )


# Trapezoid Rule
def trapezoid(f: Callable, a: float, b: float):
    return (
            (b - a) * 0.5 * (f(a) - f(b))
    )

# Composite Trapezoidal rule
# see quadrature/trapezoidal

# Composite Simpsons rule
# see quadrature/composite_simpson
