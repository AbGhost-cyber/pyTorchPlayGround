import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import lu_factor, lu_solve, expm

# Define the matrix
A = np.array([[0, -2, -2], [0, 2, -3], [0, 0, 2]])

# Calculate the 2-norm using np.linalg.norm
norm_A = np.linalg.eigvals(A.T @ A)

# Print the result
#print(norm_A)

# # calculate infinity norm:
# infinity_norm = np.linalg.norm(A, ord=np.inf)
#
# # Print the infinity norm
# # print(infinity_norm)
#
# number 2:
# Define matrix A and vector b
A = np.array([[2, 1, 2], [5, -1, 1], [1, -3, -4]])
b = np.array([5, 8, -4])

# Perform LU decomposition with pivoting
LU, piv = lu_factor(A)

# Solve the system of equations
x = lu_solve((LU, piv), b)






#
#
# print("The solution of the system Ax=b is: ", x)
#
#
# # question 3:
# # Define the coefficient matrix
# A = np.array([[1, -2, -1], [1, -1, 1], [1, 0, -1]])
#
# # Find the eigenvectors and eigenvalues
# eigvals, eigvecs = np.linalg.eig(A)
#
# # Diagonalize the coefficient matrix
# S = eigvecs
# S_inv = np.linalg.inv(S)
# D = np.diag(eigvals)
# A_diag = S @ D @ S_inv
#
# # Define the initial conditions and time points
# x0 = np.array([1, 1, 1])
# t_span = [0, 10]
# t_eval = np.linspace(t_span[0], t_span[1], 1000)
#
# # Find the solution using diagonalization and matrix exponential
# x_t = np.zeros([3, len(t_eval)])
# for i, t in enumerate(t_eval):
#     expDt = expm(D * t)
#     x_t[:, i] = S @ expDt @ S_inv @ x0
#
# # Plot the solution
# labels = ['x1', 'x2', 'x3']
#
# # Print the solution
# for i, label in enumerate(labels):
#     print(f'{label}: {x_t[i, -1]}')
# for i in range(3):
#     plt.plot(t_eval, x_t[i], label=labels[i])
# plt.xlabel('Time')
# plt.ylabel('State')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    print()
