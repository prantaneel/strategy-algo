import numpy as np
import matplotlib.pyplot as plt
import tqdm
from generate_topology import *
from generate_combination_policy import *
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
        sigma_u2_vec = np.random.rand(M, 1) + 1
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
    trace_values[k] = np.sum(RU[:, k])
    #for each agent k, this computes the Trace of R_{u,k}

b = np.zeros((N, 1))

if type_policy == 'uniform':
    b = np.ones((N, 1))
    

#############################################
# Generating the topology
#############################################
parameter = radius
algebraic_connectivity = 0
while algebraic_connectivity < 1e-4:
    adjacency,laplacian,algebraic_connectivity,degree,coordinates = generate_topology(N, parameter)

color = np.zeros((N, 1))
plot_topology(adjacency, coordinates, color)

A, p = generate_combination_policy(adjacency, b, type_policy)
#A is combination policy, p is its Perron eigenvector

#############################################
# Finding the limit point w_star
#############################################

waux = np.zeros(M) # auxiliary Mx1 vector
baux = np.zeros(M) # auxiliary Mx1 vector
if same_model == 1: # all agents have the same model wo
    for k in range(1,N+1): # finding the limit point w_star in the MSE network case
        xd = mu[k-1]*p[k-1]*RU[:,k-1]
        waux = waux + np.diag(xd)*wo  
        baux = baux + xd
else: # agents have different models w_{k}^o
    for k in range(1,N+1): # finding the Pareto solution w_star in the MSE network case
        xd = mu[k-1]*p[k-1]*RU[:,k-1]
        waux = waux + np.diag(xd).dot(Wo[:,k-1])
        print(np.diag(xd).dot(Wo[:,k-1]).shape, waux.shape)
        baux = baux + xd
w_star = np.linalg.inv(np.diag(baux))*waux # limit point

#############################################
# Running the experiments to generate the learning curves
#############################################   

MSD_av = np.zeros((1, Num_iter)) #average MSD curve for the ATC network
MSD_agent = np.zeros((N, Num_iter)) #each row contains the MSD curve for the corresponding agent in the ATC network

MSD_av_CTA = np.zeros((1, Num_iter))
MSD_agent_CTA = np.zeros((N, Num_iter))

MSD_av_CON = np.zeros()


MSD_av = np.zeros(Num_iter)   # average MSD curve for the ATC network.
MSD_agent = np.zeros((N, Num_iter))   # each row contains the MSD curve for the corresponding agent in the ATC network
MSD_av_CTA = np.zeros(Num_iter)   # average MSD curve for the CTA network.
MSD_agent_CTA = np.zeros((N, Num_iter))   # each row contains the MSD curve for the corresponding agent in the CTA network
MSD_av_CON = np.zeros(Num_iter)   # average MSD curve for the consensus network.
MSD_agent_CON = np.zeros((N, Num_iter))   # each row contains the MSD curve for the corresponding agent in the consensus network
MSD_av_NCOP = np.zeros(Num_iter)   # average MSD curve for the non-cooperative solution over N agents.
MSD_agent_NCOP = np.zeros((N, Num_iter))   # each row contains the MSD curve for the corresponding agent in the non-cooperative network
MSD_CEN = np.zeros(Num_iter)   # MSD curve for the centralized solution over N agents.
MSD_WCEN = np.zeros(Num_iter)   # MSD curve for the weighted centralized solution over N agents.
wb = tqdm.tqdm(total=Num_iter, desc='Simulating...Please wait')

for L in range(Num_trial):
    
    #ATC initialization
    psi = np.zeros((M, N)) #psi column vector for all the agents in the ATC network
    w = np.zeros((M, N)) #iterate column vectors for all the agents in the ATC network
    tilde_w = np.zeros((M, N)) #error column vector for all agents in the ATC network
    
    # CTA initialization
    psi_CTA = np.zeros((M,N))       # psi column vectors for all agents in the CTA network
    w_CTA   = np.zeros((M,N))       # iterate column vectors for all agents in the CTA network
    tilde_w_CTA = np.zeros((M,N))   # error column vectors for all agents in the CTA network
    # consensus initialization
    psi_CON = np.zeros((M,N))       # psi column vectors for all agents in the consensus network
    w_CON   = np.zeros((M,N))       # iterate column vectors for all agents in the consensus network
    tilde_w_CON  = np.zeros((M,N))  # error column vectors for all agents in the consensus network
    # non-cooperative solution initialization
    w_NCOP   = np.zeros((M,N))           # iterate column vectors for all agents in the non-cooperative solution
    tilde_w_NCOP   = np.zeros((M,N))     # error column vectors for all agents in the non-cooperative solution
    # centralized solution initialization
    w_CEN   = np.zeros((M,1))           # iterate column vectors for the centralized solution
    tilde_w_CEN   = np.zeros((M,1))     # error column vector for the centralized solution
    # weighted centralized solution initialization
    w_WCEN  = np.zeros((M,1))           # iterate column vectors for the weighted centralized solution
    tilde_w_WCEN  = np.zeros((M,1))     # error column vector for the weighted centralized solution

    for i in range(Num_iter):
        tqdm.tqdm(((L-1)*Num_iter+i)/(Num_iter*Num_trial),wb)
        #iterating over time
        
        #CTA
        for k in range(N):
            psi_CTA[:, k] = np.zeros((M, 1))
            for l in range(N):
                psi_CTA[:, k] = psi_CTA[:, k] + A[l, k]*w_CTA[:, l]
        
        #consensus
        for k in range(N):
            psi_CON[:, k] = np.zeros((M, 1))
            for l in range(N):
                psi_CON[:, k] = psi_CON[:, k] + A[l, k]*w_CON[:, l]
        
        sum_gradients = np.zeros((M, 1))
        #used by the centralized solution to accumulate the inst. gradients
        
        sum_W_gradients = np.zeros((M, 1))
        #used by weighted centralized solution to accumulate the instantaneous gradients
        
        for k in range(N): #generate data for each agent at time i
            if same_model == 1:
                if h == 1:
                    uk = np.random.randn(1, M) * np.diag(sqRU[:, k])
                    dk = uk.dot(wo) + np.random.randn() * np.sqrt(sigma_v2[k])
                else:
                    uk = np.random.randn(1, M) * np.diag(sqRU[:, k])
                    dk = uk.dot(wo) + np.random.randn(1, 1) * np.sqrt(sigma_v2[k])
            else:
                if h == 1:
                    uk = np.random.randn(1, M) * np.diag(sqRU[:, k])
                    dk = uk.dot(Wo[:, k]) + np.random.randn() * np.sqrt(sigma_v2[k])
                else:
                    uk = np.random.randn(1, M) * np.diag(sqRU[:, k])
                    dk = uk.dot(Wo[:, k]) + np.random.randn(1, 1) * np.sqrt(sigma_v2[k])
            
            psi[:, k] = w[:, k] + (2/h)*mu[k]*np.dot(uk.conj().T, (dk - np.dot(uk, w[:, k]))) #ATC diffusion (Adaptation step)
            w_CTA[:, k] = psi_CTA[:, k] + (2/h)*mu[k]*np.dot(uk.conj().T, (dk - np.dot(uk, psi_CTA[:, k]))) #CTA diffusion adaptation step
            
            w_CON[:, k] = psi_CON[:, k] + (2/h)*mu[k]*np.dot(uk.conj().T, (dk - np.dot(uk, w_CON[:, k]))) #consensus (adaptation step)
            
            w_NCOP[:, k] = w_NCOP[:, k] + (2/h)*mu[k]*np.dot(uk.conj().T, (dk - np.dot(uk, w_CON[:, k]))) #non-cooperative adaptation step
            
            sum_gradients = sum_gradients + (2 / (h*N))*mu_max*np.dot(uk.conj().T, (dk - np.dot(uk, w_CEN))) #centralized solution
            
            #central solution uses mu_max/N as step-size
            sum_W_gradients = sum_W_gradients + (2/h)*mu_max*p[k]*np.dot(uk.conj().T, (dk - np.dot(uk, w_WCEN)))
            
    w_CEN = w_CEN + sum_gradients
    tilde_w_CEN = wo - w_CEN
    
    MSD_CEN[i] = MSD_CEN[i] + (np.linalg.norm(tilde_w_CEN, 2))**2
    
    w_WCEN = w_WCEN + sum_W_gradients
    tilde_w_CEN = wo - w_WCEN
    
    MSD_WCEN[i] = MSD_WCEN[i] + (np.linalg.norm(tilde_w_WCEN, 2))**2
    
    for k in range(N):
        w[:, k] = np.zeros((M, 1))
        for l in range((M, 1)):
            w[:, k] = w[:, k] + A[l, k]*psi[:, l]
        
        #ATC        
        tilde_w[:, k] = w_star - w[:, k]
        MSD_agent[k, i] = MSD_agent[k, i] + (np.linalg.norm(tilde_w[:, k], 2))**2
        
        #CTA
        tilde_w_CTA[:, k] = w_star - w_CTA[:, k]
        MSD_agent_CTA[k, i] = MSD_agent_CTA[k, i] + (np.linalg.norm(tilde_w_CTA[:, k], 2))**2
        
        #Consensus
        tilde_w_CTA[:, k] = w_star - w_CON[:, k]
        MSD_agent_CON[k, i] = MSD_agent_CON[k, i] + (np.linalg.norm(tilde_w_CON[:, k], 2))**2
        
        #Non-coop
        tilde_w_NCOP[:, k] = wo - w_NCOP[:, k]
        MSD_agent_NCOP[k, i] = MSD_agent_NCOP[k, i] + (np.linalg.norm(tilde_w_NCOP[:, k], 2))**2

# ATC network learning curve
MSD_agent = MSD_agent / Num_trial  # each row contains the MSD evolution of the corresponding agent
MSD_av = sum(MSD_agent) / N  # average MSD evolution of the network
MSD_av_db = 10 * math.log10(MSD_av)  # dB curve

# CTA network learning curve
MSD_agent_CTA = MSD_agent_CTA / Num_trial  # each row contains the MSD evolution of the corresponding agent
MSD_av_CTA = sum(MSD_agent_CTA) / N  # average MSD evolution of the network
MSD_av_db_CTA = 10 * math.log10(MSD_av_CTA)  # dB curve

# consensus network learning curve
MSD_agent_CON = MSD_agent_CON / Num_trial  # each row contains the MSD evolution of the corresponding agent
MSD_av_CON = sum(MSD_agent_CON) / N  # average MSD evolution of the network
MSD_av_db_CON = 10 * math.log10(MSD_av_CON)  # dB curve

# non-cooperative solution learning curve
MSD_agent_NCOP = MSD_agent_NCOP / Num_trial  # each row contains the MSD evolution of the corresponding agent
MSD_agent_db_NCOP = 10 * math.log10(MSD_agent_NCOP)  # dB value
MSD_av_NCOP = sum(MSD_agent_NCOP) / N  # average MSD evolution of the non-cooperative solution
MSD_av_db_NCOP = 10 * math.log10(MSD_av_NCOP)  # dB curve

# centralized learning curve
MSD_CEN = MSD_CEN / Num_trial
MSD_CEN_db = 10 * math.log10(MSD_CEN)  # dB curve

# weighted centralized learning curve
MSD_WCEN = MSD_WCEN / Num_trial
MSD_WCEN_db = 10 * math.log10(MSD_WCEN)  # dB curve

#############################################
# Theoretical performance levels via two methods
#############################################   
# using low-rank approximation formula (provides good approximation for small step-sizes)
import numpy as np

idx = np.random.choice(N, 2, replace=False)
idx1 = idx[0]
idx2 = idx[1]
if h == 1: # real data
    S1 = np.zeros((M, M)) # used by distributed solution
    S2 = np.zeros((M, M)) # used by distributed solution
    S3 = np.zeros((M, M)) # used by non-cooperative solution
    S4 = np.zeros((M, M)) # used by centralized solution
    S5 = np.zeros((M, M)) # used by centralized solution
    Hk = np.zeros((M, M))
    Gk = np.zeros((M, M))
    Hcal = []
    Scal = []
    for k in range(N):
        Ruk = np.diag(RU[:,k])      # R_{u,k} at agent k
        Hk = 2*Ruk               # Hessin matrix H_k = 2 R_{u,k}
        zk = Wo[:,k] - w_star    # perturbation between model w_k^o at agent k and w_star
        Wk = np.outer(zk, zk)              # rank-one weighting matrix
        Rsk = 4*sigma_v2[k]*Ruk + 4*Ruk*np.trace(Wk*Ruk) + 4*Ruk*Wk*Ruk # R_{s,k}=4*sigma_{v,k}^2*R_{u,k}+4R_{u,k}Tr(W_k R_{u,k}+4 R_{u,k}W_{k}R_{u,k}
        Gk = Rsk
        S1 = S1 + mu[k]*p[k]*Hk
        S2 = S2 + mu[k]*mu[k]*p[k]*p[k]*Gk  
        S3 = S3 + mu[k]*np.linalg.inv(Hk)*Gk  # for non-cooperarive solution (in this case, zk=0 and Gk reduces to Gk = 4*sigmav2(k)*Ruk)
        S4 = S4 + Hk  # for centralized solution
        S5 = S5 + Gk  # for centralized solution  (in this case, zk=0 and Gk reduces to Gk = 4*sigmav2(k)*Ruk)
        Hcal.append(Hk)
        Scal.append(Gk)
        if k == idx1: # for non-cooperative solution
            msdA =  (mu[k]/(2*h))*np.trace(np.linalg.inv(Hk)*Gk) # MSD for non-cooperative agent of index idx1
            sigma_vA = sigma_v2[k] # corresponding noise variance.
            muA = mu[k] # step-size
            traceA = trace_values[k] # trace of covariance matrix
        if k == idx2:
            msdB =  (mu[k]/(2*h))*np.trace(np.linalg.inv(Hk)*Gk) # MSD for non-cooperative agent of index idx1
            sigma_vB = sigma_v2[k] # corresponding noise variances.
            muB = mu[k] # step-size
            traceB = trace_values[k] #trace of covariance matrix
else:  # complex data
    S1 = np.zeros((2*M, 2*M))
    S2 = np.zeros((2*M, 2*M))
    S3 = np.zeros((2*M, 2*M))
    S4 = np.zeros((2*M, 2*M)) # used by centralized solution
    S5 = np.zeros((2*M, 2*M)) # used by centralized solution
    Hk = np.zeros((2*M, 2*M))
    Gk = np.zeros((2*M, 2*M)) # off-diagonal blocks of Gk are irrelevant
    Hcal = []
    Scal = []
    for k in range(N):
        Ruk = np.diag(RU[:,k])                     # R_{u,k} at agent k
        Hk[0:M,0:M] = Ruk                       # Hessin matrix H_k = block diagonal{R_{u,k},R_{u,k}^T}
        Hk[M+1:2*M,M+1:2*M] = np.transpose(Ruk)    
        zk = Wo[:,k] - w_star  # perturbation between model w_k^o at agent k and w_star
        Wk = np.outer(zk, zk)            # rank-one weighting matrix
        Rsk = sigma_v2[k]*Ruk + Ruk*np.trace(Wk*Ruk) # R_{s,k} =sigma_{v,k}^2*R_{u,k}+R_{u,k}Tr(W_k R_{u,k}
        Gk[0:M,0:M] = Rsk      # G_k = [R_{s,k} X; X R_{s,k}^T]; X is irrelevant
        Gk[M+1:2*M,M+1:2*M] = np.transpose(Rsk)
        S1 = S1 + mu[k]*p[k]*Hk
        S2 = S2 + mu[k]*mu[k]*p[k]*p[k]*Gk  
        S3 = S3 + mu[k]*np.linalg.inv(Hk)*Gk  # for non-cooperarive solution  (in this case, zk=0 and Gk reduces to Gk = sigmav2(k)*Ruk)
        S4 = S4 + Hk # for centralized solution 
        S5 = S5 + Gk # for centralized solution  (in this case, zk=0 and Gk reduces to Gk = sigmav2(k)*Ruk)
        Hcal.append(Hk)
        Scal.append(Gk)
        if k == idx1: # for non-cooperative solution
            msdA =  (mu[k]/(2*h))*np.trace(np.linalg.inv(Hk)*Gk) # MSD for non-cooperative agent of index idx1
            sigma_vA = sigma_v2[k] # corresponding noise variances.
            traceA = trace_values[k] # trace of covariance matrix
            muA = mu[k]
        if k == idx2:
            msdB =  (mu[k]/(2*h))*np.trace(np.linalg.inv(Hk)*Gk) # MSD for non-cooperative agent of index idx1
            sigma_vB = sigma_v2[k] # corresponding noise variances.
            traceB = trace_values[k] #trace of covariance matrix
            muB = mu[k]      
MSD_thy_low_rank    = (1/(2*h))*np.trace(np.linalg.inv(S1)*S2)  # Theoretical MSD from the low-rank approximation formula
MSD_thy_low_rank_db = 10*np.log10(MSD_thy_low_rank)
MSD_thy_small_mu_NCOP    = (1/(2*h*N))*np.trace(S3)   # Theoretical MSD for small step-sizes for the non-cooperative solution
MSD_thy_small_mu_db_NCOP = 10*np.log10(MSD_thy_small_mu_NCOP)
MSD_thy_small_mu_CEN    = (mu_max/(2*h*N))*np.trace(np.linalg.inv(S4)*S5)   # Theoretical MSD for small step-sizes for centralized solution
MSD_thy_small_mu_db_CEN = 10*np.log10(MSD_thy_small_mu_CEN)
# more accurate formulas without using the low-rank approximation; the
# expressions for Bcal and Ycal change for ATC, CTA, consensus
#ATC
Ao = np.eye(N,N)
A1 = np.eye(N,N)
A2 = A
Mcal  = np.kron(np.diag(mu),np.eye(h*M,h*M))
Aocal = np.kron(Ao,np.eye(h*M,h*M))
A1cal = np.kron(A1,np.eye(h*M,h*M))
A2cal = np.kron(A2,np.eye(h*M,h*M))
Bcal  = (A2cal.T)@((Aocal.T)-Mcal@Hcal)@(A1cal.T)
Ycal  = (A2cal.T)@Mcal@Scal@Mcal@A2cal
Xcal = np.eye(h*M*N,h*M*N)
MSD_thy = 0
for n in range(2500):  # using the series formula rather than the closed-form expression to avoid out-of-memory message
    MSD_thy = MSD_thy + np.trace(Xcal@Ycal@(Xcal.T))   
    Xcal = Xcal@Bcal
MSD_thy = (1/(h*N))*MSD_thy
MSD_thy_db = 10*np.log10(MSD_thy)
## if we were to use the closed-form expression; replace the above for loop by the following calculation (or use a Lyapunov equation instead).
# Fcal  = np.kron(np.transpose(Bcal),(Bcal))  # Fcal = B^T \otimes B^*
# r = np.reshape(np.eye(h*M*N,h*M*N),[],1)    # r = vec(I_{hMN});
# y = np.reshape(Ycal.T,[],1)               # y = vec(Ycal^T);
# L = (h*M*N)**2;
# sigma= (np.eye(L,L)-Fcal)\y;               # (I-Fcal)^{-1}y
# MSD_thy = (1/(h*N))*y.T@sigma
# MSD_thy_db = 10*np.log10(MSD_thy)
##
# CTA
AoCTA = np.eye(N,N)
A1CTA = A
A2CTA = np.eye(N,N)
Mcal  = np.kron(np.diag(mu),np.eye(h*M,h*M))
AocalCTA = np.kron(AoCTA,np.eye(h*M,h*M))
A1calCTA = np.kron(A1CTA,np.eye(h*M,h*M))
A2calCTA = np.kron(A2CTA,np.eye(h*M,h*M))
BcalCTA  = (A2calCTA.T)@((AocalCTA.T)-Mcal@Hcal)@(A1calCTA.T)
YcalCTA  = (A2calCTA.T)@Mcal@Scal@Mcal@A2calCTA
Xcal = np.eye(h*M*N,h*M*N)
MSD_thy_CTA = 0
for n in range(2500):   # using the series formula rather than the closed-form expression to avoid out-of-memory message
    MSD_thy_CTA = MSD_thy_CTA + np.trace(Xcal@YcalCTA@(Xcal.T))   
    Xcal = Xcal@BcalCTA
MSD_thy_CTA = (1/(h*N))*MSD_thy_CTA
MSD_thy_db_CTA = 10*np.log10(MSD_thy_CTA)
# consensus
AoCON = A
A1CON = np.eye(N,N)
A2CON = np.eye(N,N)
Mcal  = np.kron(np.diag(mu),np.eye(h*M,h*M))
AocalCON = np.kron(AoCON,np.eye(h*M,h*M))
A1calCON = np.kron(A1CON,np.eye(h*M,h*M))
A2calCON = np.kron(A2CON,np.eye(h*M,h*M))
BcalCON  = (A2calCON.T)@((AocalCON.T)-Mcal@Hcal)@(A1calCON.T)
YcalCON  = (A2calCON.T)@Mcal@Scal@Mcal@A2calCON
Xcal = np.eye(h*M*N,h*M*N)
MSD_thy_CON = 0
for n in range(2500):   # using the series formula rather than the closed-form expression to avoid out-of-memory message
    MSD_thy_CON = MSD_thy_CON + np.trace(Xcal@YcalCON@(Xcal.T))   
    Xcal = Xcal@BcalCON
MSD_thy_CON = (1/(h*N))*MSD_thy_CON
MSD_thy_db_CON = 10*np.log10(MSD_thy_CON)
    
#############################################
# Generating figures
#############################################   

if same_model == 1: # when all agents have the same model wo, we plot curves for ATC, CTA, consensus, non-cooperative, and centralized
    
    # distributed solutions
    iter = np.arange(1, Num_iter+1)
    plt.figure()
    plt.plot(iter, MSD_av_db, 'b-', iter, MSD_av_db_CTA, 'r-', iter, MSD_av_db_CON, 'k-', iter, MSD_CEN_db, 'g-', iter, MSD_WCEN_db, 'y-', iter, MSD_av_db_NCOP, 'm-', linewidth=1)
    plt.hold(True)
    plt.plot(iter, MSD_thy_low_rank_db*np.ones(Num_iter), 'b--', iter, MSD_thy_small_mu_db_CEN*np.ones(Num_iter), 'g-', iter, MSD_thy_low_rank_db*np.ones(Num_iter), 'y--', iter, MSD_thy_small_mu_db_NCOP*np.ones(Num_iter), 'm--', linewidth=2)
    plt.plot(iter, MSD_thy_db*np.ones(Num_iter), 'b:', iter, MSD_thy_db_CTA*np.ones(Num_iter), 'r:', iter, MSD_thy_db_CON*np.ones(Num_iter), 'k:', linewidth=1)
    plt.title('$$N=$$' + str(N) + '$$,M=$$' + str(M), fontsize=10)
    plt.grid(True)
    plt.xlabel('$$i$$ (iteration)', fontsize=12)
    plt.ylabel('$$\textrm{MSD}_{\textrm{dist,av}}(i)$$ (dB)', fontsize=12)
    plt.axis([0, Num_iter, -60, 0])
    plt.legend(['ATC', 'CTA', 'consensus', 'centralized', 'weigthed centralized', 'non-cooperative', 'theory small \mu', 'theory centralized', 'theory weighted cen', 'theory NCOP', 'theory ATC', 'theory CTA', 'theory consensus'])
    
    # non-cooperative solution
    plt.figure()
    plt.plot(iter, MSD_av_db_NCOP, 'b-', iter, MSD_agent_db_NCOP[idx1,:], 'r-', iter, MSD_agent_db_NCOP[idx2,:], 'k-', linewidth=1)
    plt.hold(True)
    plt.plot(iter, MSD_thy_small_mu_db_NCOP*np.ones(Num_iter), 'b--', linewidth=2)
    plt.plot(iter, 10*np.log10(msdA)*np.ones(Num_iter), 'b:', iter, 10*np.log10(msdB)*np.ones(Num_iter), 'r:', linewidth=1)
    plt.title('$$N=$$' + str(N) + '$$,M=$$' + str(M), fontsize=10)
    plt.grid(True)
    plt.xlabel('$$i$$ (iteration)', fontsize=12)
    plt.ylabel('$$\textrm{MSD}_{\textrm{ncop,av}}(i)$$ (dB)', fontsize=12)
    plt.axis([0, Num_iter, -60, 0])
    plt.legend(['non-cooperative', 'agent A', 'agent B', 'theory NCOP', 'theory A', 'theory B'])
    
    print('indices of the two non-cooperative agents')
    print([idx1, idx2])
    print('their step-sizes')
    print([muA, muB])
    print('their noise variances')
    print([sigma_vA, sigma_vB])
    print('trace of their regression covariance matrices')
    print([traceA, traceB])
else: # different models w_k^o; in this case, it is not meaningful to plot the non-cooperative and centralized curves
    import matplotlib.pyplot as plt
    import numpy as np
    
    # distributed solutions
    iter = np.arange(1, Num_iter+1)
    plt.figure()
    plt.plot(iter, MSD_av_db, 'b-', iter, MSD_av_db_CTA, 'r-', iter, MSD_av_db_CON, 'k-', linewidth=1)
    plt.hold(True)
    plt.plot(iter, MSD_thy_low_rank_db*np.ones(Num_iter), 'b--', linewidth=2)
    plt.plot(iter, MSD_thy_db*np.ones(Num_iter), 'b:', iter, MSD_thy_db_CTA*np.ones(Num_iter), 'r:', iter, MSD_thy_db_CON*np.ones(Num_iter), 'k:', linewidth=1)
    plt.title('$$N=$$' + str(N) + '$$,M=$$' + str(M), fontsize=10)
    plt.grid(True)
    plt.xlabel('$$i$$ (iteration)', fontsize=12)
    plt.ylabel('$$\textrm{MSD}_{\textrm{dist,av}}(i)$$ (dB)', fontsize=12)
    plt.axis([0, Num_iter, -60, 0])
    plt.legend(['ATC', 'CTA', 'consensus', 'theory small \mu', 'theory ATC', 'theory CTA', 'theory consensus'])
    
plt.figure() # regression power and noise power
plt.subplot(221)
plt.plot(np.arange(1, N+1), 10*np.log10(trace_values), '-o', linewidth=1, markersize=4, markeredgecolor='k', markerfacecolor='g')
plt.grid(True)
plt.xlabel('$$k$$ (index of node)', fontsize=10)
plt.ylabel('$$\textrm{Tr}(R_{u,k})$$ (dB)', fontsize=10)
plt.axis([1, N, 9, 13])
plt.title('$$\textrm{power of regression data}$$', fontsize=12)
plt.subplot(222)
plt.plot(np.arange(1, N+1), 10*np.log10(sigma_v2), '-o', linewidth=1, markersize=4, markeredgecolor='k', markerfacecolor='r')
plt.grid(True)
plt.xlabel('$$k$$ (index of node)', fontsize=10)
plt.ylabel('$$\sigma_{v,k}^2$$ (dB)', fontsize=10)
plt.axis([1, N, -25, -15])
plt.title('$$\textrm{noise power}$$', fontsize=12)
plt.show()