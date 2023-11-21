import numpy as np
import matplotlib.pyplot as plt


def generate_topology(N, parameter):
    
    A = np.zeros((N, N))
    L = np.zeros((N, N))
    r = parameter
    x_coordinates = np.random.rand(1, N) + 0.1
    y_coordinates = np.random.rand(1, N) + 0.1
    Coordinates = np.column_stack((x_coordinates.T, y_coordinates.T))
    
    for k in range(N):
        for l in range(N):
            d = np.sqrt((x_coordinates[0, k] - x_coordinates[0, l])**2 + (y_coordinates[0, k] - y_coordinates[0, l])**2)
            if d <= r:
                A[k, l] = 1  # set entry in adjacency matrix to one if nodes k and l should be neighbors.

    Adjacency = A
    
    # We determine the number of neighbors of each node from the adjacency matrix
    num_nb = np.zeros(N)
    for k in range(N):
        num_nb[k] = np.sum(A[k, :])

    Degree_Vector = num_nb  # vector of degrees for the various nodes
    
    for k in range(N):
        L[k, k] = max(0, np.sum(A[k, :]) - 1) # set diagonal entry to zero if degree-1 for node k is negative.
        for l in range(k + 1, N):
            L[k, l] = -1 * A[k, l]
            L[l, k] = -1 * A[l, k]

    sigma = np.linalg.svd(L)[1] # vector of singular values of L.
    Laplacian = L #Laplacian matrix
    Algebraic_Connectivity = sigma[N-2] # algebraic connectivity
    if np.sum(sigma[:-1]) < 1e-4:
        return
    
    return Adjacency, Laplacian, Algebraic_Connectivity, Degree_Vector, Coordinates

def plot_topology(Adjacency, Coordinates, Color):
    A = Adjacency  # adjacency matrix
    N = len(A)  # number of agents
    x_coordinates = Coordinates[:, 0]  # x-coordinates of agents
    y_coordinates = Coordinates[:, 1]  # y-coordinates of agents

    plt.figure()

    for k in range(N):
        for l in range(N):
            if A[k, l] > 0:
                plt.plot([x_coordinates[k], x_coordinates[l]], [y_coordinates[k], y_coordinates[l]], 'b-', linewidth=1.5)

    for k in range(N):
        if Color[k] == 0:  # yellow
            plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='y', markersize=10)
        else:
            if Color[k] == 1:  # red
                plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='r', markersize=10)
            else:  # green
                plt.plot(x_coordinates[k], y_coordinates[k], 'o', markeredgecolor='b', markerfacecolor='g', markersize=10)

    plt.axis([0, 1.2, 0, 1.2])
    plt.axis('square')
    plt.grid(True)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    for k in range(N):
        plt.text(x_coordinates[k] + 0.03, y_coordinates[k] + 0.03, str(k), fontsize=7)
    
    plt.show()



    
algebraic_connectivity = 0;
while algebraic_connectivity < 1e-4:
   adjacency,laplacian,algebraic_connectivity,degree,coordinates = generate_topology(20,0.3)

color = np.zeros(20)
plot_topology(adjacency, coordinates, color)


