'''
Node vector is a regressor vector considering the past w_opt values
u(t), u(t-1), u(t-2), ...
Initialisation with all the values being zero


'''


import numpy as np
import tqdm
x = np.ones((2, 1))
y = np.ones((2, 3))
x_coordinates = np.random.rand(1, 10) + 0.1
a = np.ones((1, 10))
b = np.ones((10, 1))
print(a.dot(b))
