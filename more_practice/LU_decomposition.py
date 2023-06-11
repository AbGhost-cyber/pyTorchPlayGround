import numpy as np
import scipy

# This code demonstrates how to solve the problem of Question 8a
# using the direct method(LU decomposition)

# Set the value of n
n = 6

# Generate the matrix A and vector b (aij = 1/i + j - 1)
A = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])
x = np.ones(n)
# b is the dot product where b = Ax
b = np.dot(A, x)

# Solve for x using LU decomposition
lu, piv = scipy.linalg.lu_factor(A)
x = scipy.linalg.lu_solve((lu, piv), b)

# Calculate the errors
b_h = np.dot(A, x)  # Calculate predicted b using the solved x
b_error = np.linalg.norm(b - b_h, 2)  # Calculate L2 norm of difference between b and predicted b
x_error = np.linalg.norm(x - np.ones(n), 2)  # Calculate L2 norm of difference between x and expected x

# Calculate x_bar
x_bar = np.ones(n)

if __name__ == '__main__':
    # Print the results
    print('x: ', x_bar)
    print('||b - Ax||_2: ', b_error)
    print('||x - x̄||_2: ', x_error)
