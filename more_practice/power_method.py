import numpy as np
import scipy
from scipy.linalg import norm

# This code demonstrates how to solve the problem of Question 8b

# Define value of n
n = 5

# Define random matrix B and construct matrix A
B = np.random.rand(n, n)
A = B + B.T

# Define initial vector
u = np.ones(n)

# Define stop criterion 10^-6
eps = 1e-6

# Define maximum number of iterations
max_iter = 1000

# Initialize iteration counter
k = 0

# Start power iteration
while k < max_iter:
    v = A @ u
    lam = np.abs(v).max()
    v = v / lam
    if norm(v - u) < eps:
        break
    u = v
    k += 1
# Find corresponding eigenvector
eigv = u / norm(u)

# Prints the iteration number, the result of eigenvalue and corresponding eigenvector
print(f"Iterations: {k}")
print(f"Eigenvalue: {lam}")
print(f"Eigenvector: {eigv}")

if __name__ == '__main__':
    print()