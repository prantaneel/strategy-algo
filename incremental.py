import numpy as np
def updateNodeWeightEstimateIncremental(node_id, iter, step_size, node_data, node_vector, w_opt_size, prev_node_weight_est):
    node_weight_est = np.ones((w_opt_size, 1))
    p_prev = prev_node_weight_est
    node_weight_est = prev_node_weight_est + step_size*np.dot(node_vector, (node_data - np.dot(node_vector.conj().T, p_prev)))
    return node_weight_est

