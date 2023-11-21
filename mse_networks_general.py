import numpy as np
"""
Only distributed learning so same_model = 0
"""
h = 1 #only for real data change to 2 for complex data

uniform_step = 1

same_model = 0
uniform_reg = 1
"""
% set it to the following values:
% 0: (different) Each R_{u,k} is a diagonal matrix with possibly different diagonal entries chosen randomly
% 1: (different but white) Each R_{u,k} is of the form R_{u,k} = \sigma_{u,k}^2 I_{M}
% 2: (uniform) All R_{u,k} equal to each other and of the form R_{u,k} = \sigma_{u}^2 I_M
"""
mu_max = 0.001 #max step-size param used for adapting

N = 20 #number of agents in the network
M = 10 #size of the wo, which is Mx1
Num_iter = 5000 #number of iterations per experiment
Num_trial = 100 #number of experiments

type_policy = 'uniform'

radius = 0.3

#############################################
# Generating the signal and the noise powers
#############################################

if h == 1:
    #real data
    wo = np.random.randn(M, 1)
    wo = wo / np.linalg.norm(wo, 2)
else:
    wo = np.random.randn(M, 1)+1j*np.random.randn(M, 1)
    wo = wo / np.linalg.norm(wo, 2)

if same_model == 1:
    for k in range(N):
        Wo[:, k] = wo

if same_model == 0:
    #agents use different models
    if h == 1:
        Wo = np.random.randn(M, N)
    else:
        Wo = np.random.randn(M, N)+1j*np.random.randn(M, N)
    
    for k in range(N):
        #separate normalisation for each agent
        Wo[:, k] = Wo[:, k] / np.linalg.norm(Wo[:, k], 2)

if uniform_step == 0:
    mu = mu_max*(0.5*np.random.rand(N, 1) + 0.5)
    #generated the step-sizes for all agents by randomizing it
else:
    mu = mu_max*np.ones((N, 1))

sigma_v2_dB = np.random.rand(N, 1)*10 - 25 #noise power between -25 and -15dB
sigma_v2 = (10.0)**(sigma_v2_dB / 10)
"""
% Each R_{u,k} is diagonal of size MxM.
% We represent each R_{u,k} by an Mx1 column vector representing its diagonal. 
% We collect all diagonal vectors into an MxN matrix RU. 
"""
RU = np.zeros((M, N))
sqRU = np.zeros((M, N)) #contains the square-root of the entries on each column of RU

if uniform_reg == 0:
    for k in range(N):
        sigma_u2_vec = np.random.rand(M) + 1
        RU[:,k] = sigma_u2_vec
        sqRU[:,k] = np.sqrt(RU[:,k])

if uniform_reg == 1:
    for k in range(N):
        sigma_u2 = np.random.rand() + 1
        RU[:,k] = sigma_u2*np.ones(M)
        sqRU[:,k] = np.sqrt(RU[:,k])

if uniform_reg == 2:
    sigma_u2 = np.random.rand() + 1
    RU = sigma_u2*np.ones((M,N))
    sqRU = np.sqrt(sigma_u2)*np.ones((M,N))

trace_values = np.zeros((N, 1))
for k in range(N):
    

    