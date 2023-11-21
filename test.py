'''
Node vector is a regressor vector considering the past w_opt values
u(t), u(t-1), u(t-2), ...
Initialisation with all the values being zero


'''


import numpy as np

x = np.ones((2, 1))
y = np.ones((2, 3))
x_coordinates = np.random.rand(1, 10) + 0.1
x[1] = 2
print(np.random.randn(10, 1)) 