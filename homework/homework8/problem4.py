import numpy as np
import numpy.linalg.linalg

A = np.array([[1, 0],
              [0, 1],
              [0, 1]])

b = np.array([[1],
              [1],
              [0]])

# want to solve A^TAx = A^Tb
A_p = A.T @ A
b_p = A.T @ b

x, residuals, rank, s = numpy.linalg.linalg.lstsq(a=A_p, b=b_p, rcond=None)
print(x)

# yields:
# [[1. ]
#  [0.5]]
