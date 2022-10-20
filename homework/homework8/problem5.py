import numpy as np

A = np.array([[1, 3],
              [6, -1],
              [4, 0],
              [2, 7]])

D = np.zeros((4, 4))
D[0][0] = 1
D[1][1] = 2
D[2][2] = 5
D[3][3] = 3

C = np.array([[1],
              [2],
              [3],
              [4]])

A_tilde = D @ A

b = np.array([[.0975],
              [0.278],
              [0.54688],
              [0.9575]])

b_tilde = D @ (b + C)
x = np.linalg.inv((A_tilde.T @ A_tilde)) @ A_tilde.T @ b_tilde

print(x)
print()
print(A @ x - C)
