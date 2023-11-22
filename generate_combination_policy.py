"""
Combination policy in diffusion strategy is used to combine the psi from the neighbourhood using some coefficients.
"""
import numpy as np
def generate_combination_policy(Adjacency, b, type='uniform'):
    """
    by default type is uniform and generates uniform combination
    b = ones(N, 1)
    b: auxiliary column vector with the same number of entries as the number of nodes.
    """
    
    A = Adjacency
    N = max(A.shape)
    
    #we need to determine the number of neighbours of each node from the adjacency matrix
    num_nb = np.zeros((N, 1))
    for k in range(N):
        num_nb[k] = np.sum(A[k, :])
    type = type.lower()
    if type == 'uniform':
        W = A
        for k in range(N):
            W[k, :] = W[k, :] / np.sum(W[k, :])
    
    Combination_Matrix = np.transpose(W)  # The desired combination matrix is the transpose of W
    # Printing these two values to check that rows or columns add up to one
    # np.ones((1, N)).dot(Combination_Matrix)  # result is all ones for left-stochastic
    # np.transpose(Combination_Matrix).dot(np.ones((N, 1)))  # result is all ones for doubly stochastic
    # Finding the Perron eigenvector
    D, V = np.linalg.eig(Combination_Matrix)  # eigenvalue decomposition
    ##DEBUG: Order of return for eigenvalues and eigenvectors
    print(np.abs(np.diag(D)))
    idx = np.argmax(np.abs(np.diag(D)))  # returns index of maximum magnitude eigenvalue of A, which is the eigenvalue at one.
    idx = idx//N
    p = V[:, idx]  # extracting the corresponding eigenvector
    p = p / np.sum(p)  # normalizing the sum of its entries to one
    p_Vector = p  # Perron eigenvector
    
    return Combination_Matrix, p_Vector
             
    
    