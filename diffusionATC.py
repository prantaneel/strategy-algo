import numpy as np


def getIntermediateWeightDiffusionATC(coeff_Nk, size_Nk, step_size, node_data, node_vector, w_opt_size, num_neighbours, w_ki_1):
    psi_ki = w_ki_1 + 2*step_size*(np.dot(node_vector, (node_data - np.dot(node_vector.conj().T, w_ki_1))))
    return psi_ki


def updateNodeWeightEstimateDiffusionATC(coeff_Nk, psi_Nki, size_Nk, step_size, node_data, node_vector, w_opt_size, num_neighbours, w_ki_1):
    
    #get the intermediate weight estimations for the neighbouring nodes
    w_ki = np.zeros((w_opt_size, 1))
    for _iter in range(num_neighbours):
        psi_ki_1 += coeff_Nk[_iter]*psi_Nki[_iter]
    
    return w_ki
