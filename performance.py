import numpy as np
def calculateMSE(node_id, iteration, node_weight_vector, node_data, node_vector):
    y_pred = np.dot(node_vector.conj().T, node_weight_vector)
    mse = (node_data - y_pred)**2
    return node_id, iteration, mse


def calculateEMSE(node_id, w_opt, node_data, node_vector):
    y_pred = np.dot(node_vector.conj().T, w_opt)
    emse = (node_data - y_pred)**2
    
    return node_id, emse

def calculateMSD(node_id, iteration, w_opt, node_weight_vector):
    msd_k = np.linalg.norm(w_opt - node_weight_vector)**2
    
    return node_id, iteration, msd_k

