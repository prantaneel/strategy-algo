#LMS algorithm

'''
We have the whole network topology and we need to find a cycle in the central server
Keep it simple --> we have a simple network with the nodes being the node_id of the physical sensors

'''

import numpy as np

def updateNodeWeightEstimateConsensus(coeff_Nk, weight_Nk, size_Nk, step_size, node_data, node_vector, w_opt_size, num_neighbours, w_ki_1):
    w_ki = np.zeros((w_opt_size, 1))
    for _iter in range(num_neighbours):
        w_ki += coeff_Nk[_iter]*weight_Nk[_iter]
    
    w_ki += 2*step_size*(np.dot(node_vector, (node_data - np.dot(node_vector.conj().T, w_ki_1))))
    
    return w_ki
