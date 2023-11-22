import numpy as np
import scipy as sc

# Define a 2D matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Define a 1D array
array = np.array([2, 3, 4])

# Perform matrix multiplication using numpy.dot
result = matrix.dot(array)
H = []
x = np.matrix([[1, 2], [3, 4]])
for k in range(5):
    if k == 0:
        H = sc.linalg.block_diag(x)
    else:
        H = sc.linalg.block_diag(H, x)
print(H)