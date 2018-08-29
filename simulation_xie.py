# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os
sys.path.append(os.path.abspath('..')+'/TVGL')

import numpy as np
from myfunctions import *
import random
from scipy.linalg import block_diag
import numpy.linalg as alg

p = 100
d = 3
n_vec = [50, 50, 50, 50]
n_change = 3
len_t = 50
K = 4

# S_list = [] # upper triangle
A_list = [] # Omega graph 
C_list = []
for k in range(K+1):
    gG = getGraph(p, d)
    S = gG[0]
    A = gG[1]
    A_T = getGraphatT_Shuheng(S, A, n_change)[1]
    A_T_list = [lam*A_T+(1-lam)*A for lam in np.linspace(0, 1, len_t)] # Omega
    C_T_list = [getCov(item) for item in A_T_list] # Cov
    
    A_list.append(A_T_list)
    C_list.append(C_T_list)
    
Y_list = []
for time_ix in range(len_t):
    C0 = C_list[0][time_ix]
    Y_t = []
    for k in range(1,K+1):
        Ck = C_list[k][time_ix]
        Z = np.random.multivariate_normal(mean = np.zeros(p), cov = C0, size = n_vec[k-1])
        X = np.random.multivariate_normal(mean = np.zeros(p), cov = Ck, size = n_vec[k-1])
        Y = X+Z
        Y_t.append(Y)
    Y_list.append(Y_t)    
    
Y_array = np.array(Y_list) # time class n p
t = 0
Y_t_array = np.transpose(Y_array[t], [1, 0, 2])
Y_t = np.reshape(Y_t_array, [50, 400])

S_Y = np.matmul(Y_t.T, Y_t)/n_vec[0]
S_0 = np.zeros([p, p])
for m in range(K):
    for l in [i for i in range(K) if i != m]:
        S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
        S_0 = S_0 + S_ml
S_0 = S_0/((K-1)*K)

S_K_list = []
for k in range(K):
    S_k = S_Y[k*p+np.array(range(p))[:, None], k*p+np.array(range(p))[None, :]]
    S_k = S_k - S_0
    S_K_list.append(S_k)

S_hat_list = [S_0] + S_K_list
S_pd_list = []
# closest PSD
for k in range(K+1):
    S_pd_list.append(nearestPD(S_hat_list[k]))

lam1 = 0.5
lam2 = 0.5
lam_vec = [lam1, lam2, lam2, lam2, lam2] 

#initialize
Omega_list = [cov.graph_lasso(S_pd_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 


# start
A = np.zeros([p,p])
for k in range(K):
    A = A + Omega_list[k]
A_inv = alg.inv(A)

#E
OSO = np.zeros([p,p])
for m in range(K):
    for l in range(K):
        OSO = OSO + np.matmul(np.matmul(Omega_list[m+1],S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]), Omega_list[l+1])

S_0 = A_inv + np.matmul(np.matmul(A_inv,OSO), A_inv)      

S_K_list = []
for m in range(K):
    SO = np.zeros([p,p])
    OS = np.zeros([p,p])
    for l in range(K):
        SO = SO + np.matmul(S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]], Omega_list[l+1])
        OS = OS + np.matmul(Omega_list[l+1], S_Y[l*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]])
    S_k = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]] - np.matmul(SO, A_inv) - np.matmul(A_inv, OS) + S_0
    S_K_list.append(S_k)
S_pd_list = [S_0] + S_K_list

#M
Omega_list = [cov.graph_lasso(S_pd_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 


# likelihood
A = np.zeros([p,p])
for k in range(K):
    A = A + Omega_list[k]
A_inv = alg.inv(A)
    
likelihood_K = 0
for k in range(K):
    likelihood_K = likelihood_K + np.log(alg.det(Omega_list[k+1])) - np.trace(np.matmul(S_Y[k*p+np.array(range(p))[:, None], k*p+np.array(range(p))[None, :]], Omega_list[k+1]))

tr_OSOA = 0
for m in range(K):
    for l in range(K):
        OSOA = np.matmul(np.matmul(np.matmul(Omega_list[m+1],S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]), Omega_list[l+1]), A_inv)
        tr_OSOA = tr_OSOA + np.trace(OSOA)

likelihood = likelihood_K + np.log(alg.det(Omega_list[0])) - np.log(alg.det(A)) + tr_OSOA

penalty = 0
for k in range(K+1):
    penalty = penalty + lam_vec[k]*np.sum(np.abs(np.tril(Omega_list[k], -1) + np.triu(Omega_list[k], 1)))             
        
pen_likelihood = likelihood - penalty       
print(pen_likelihood)
        
        
        
        
        
        
        
        
        
 