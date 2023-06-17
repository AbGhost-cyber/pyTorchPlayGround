import numpy as np

# This code demonstrates how to solve the problem of Question 8a
# using the direct method(LU decomposition)

# chosen n value
n = 6

A = np.zeros((n, n))
# Compute the entries of A using the formula aij = 1/i+j-1.
for i in range(1, n + 1):
    for j in range(1, n + 1):
        A[i - 1, j - 1] = 1 / (i + j - 1)

# Compute the vector b by multiplying entries equal to 1.
b = np.ones(n)

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Multiply the inverse of A by vector b to obtain the solution vector x̄.
x̄ = np.dot(A_inv, b)

# Print the solution vector x̄
print("x̄: ", x̄)

# Calculate the error between x̄ and b
error = np.linalg.norm(b - np.dot(A, x̄))
print("||b - Ax||_2:", error)

# Calculate the error between x̄ and the initial guess of x
x_guess = np.ones(n)
error_guess = np.linalg.norm(x_guess - x̄)
print("||x - x̄||_2:", error_guess)

if __name__ == '__main__':
    print()
