import matplotlib.pyplot as plt
import numpy as np

# what we want to interpolate
f = lambda x: 1.0 / (1 + np.power(10.0 * x, 2))

# max poly degree
N = 10

# make interpolation nodes
X = np.linspace(-1, 1, N + 1)

Neval = 1_000
Xeval = np.linspace(-1, 1, Neval + 1)

# build Vandermonde matrix
V = np.zeros((N + 1, N + 1))
for j in range(0, N + 1):
    V[:, j] = np.power(X, j)

# sample the function
F = f(X)
# solve for the coeffs of the monomials: Vc = f
c = np.linalg.solve(V, F)

# sample the function on finer grid
Feval = f(Xeval)

# build the interpolation operator
Veval = np.zeros((Neval + 1, N + 1))
for j in range(0, N + 1):
    Veval[:, j] = np.power(Xeval, j)

plt.plot(X, F, 'ko-')
plt.plot(X, V @ c, 'rx')
# plt.plot(Xeval, Feval @ c, 'k-')
# plt.plot(Xeval, Veval @ c, 'r--')
plt.show()

# plt.plot(X, V, '-')
# plt.show()
