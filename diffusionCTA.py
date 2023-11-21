#LMS algorithm
#calculate the MSE for each of the iteration
import numpy as np

def updateNodeWeightEstimateDiffusionCTA(coeff_Nk, weight_Nk, size_Nk, step_size, node_data, node_vector, w_opt_size, num_neighbours, w_ki_1):
    psi_ki_1 = np.zeros((w_opt_size, 1))
    for _iter in range(num_neighbours):
        psi_ki_1 += coeff_Nk[_iter]*weight_Nk[_iter]
    
    w_ki = psi_ki_1 + 2*step_size*(np.dot(node_vector, (node_data - np.dot(node_vector.conj().T, psi_ki_1))))
    
    return w_ki
