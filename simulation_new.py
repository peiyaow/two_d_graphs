# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:21:21 2018

@author: peiyao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:32:58 2018

@author: peiyao
"""

import numpy as np
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *
# from posdef import *
import random
import multiprocessing as mp

#------------------------------------------- retrieve data ---------------------------------------------------------------------
#myseed1 = int(sys.argv[1])
#myseed2 = int(sys.argv[2])
#sim_ix = int(sys.argv[3])
#
#random.seed(myseed1)
#np.random.seed(myseed2)

p = 100
d = 3
n_vec = [50, 50, 50, 50]
n = sum(n_vec)
n_change = 3
len_t = 51
K = 4
h = 5.848/np.cbrt(len_t)
width = 5
set_length = 50 # number of lambdas

A_list, O_list, C_list, Y_list = simulate_data_2dgraph() # Y_list: class time ni p
Y_list = [np.array(Y_list[i]).transpose([1, 0, 2]).reshape([n_vec[i], len_t*p]) for i in range(K)] # n t p -> each element: ni by time*p
S_Y_list = [np.matmul(Y_list[k].T, Y_list[k])/n_vec[k] for k in range(K)]

S_0_list = []
S_0_pd_list = []
for k in range(K): 
    S_Y = S_Y_list[k]
    S_0 = np.zeros([p, p]) # Sigma_0
    for m in range(len_t):
        for l in [i for i in range(len_t) if i != m]:
            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
            S_0 = S_0 + S_ml
    S_0 = S_0/((len_t-1)*len_t)
    S_0_pd = nearestPD(S_0)
    S_0_list.append(S_0)
    S_0_pd_list.append(S_0_pd)
    
S_X_list = []
for k in range(K): 
    S_Y = S_Y_list[k]
    S_0 = S_0_list[k]
    S_X_k_list = []
    for m in range(len_t):
        S_mm = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
        S_X = nearestPD(S_mm-S_0)
        S_X_k_list.append(S_X)
    S_X_list.append(S_X_k_list)

# initial input
S_0 = np.sum(np.array([S_0_pd_list[k]*n_vec[k]/n for k in range(K)]), axis = 0)
S_X_array = np.array(S_X_list).transpose([1, 0, 2, 3])

# given time
t = 25
lam1 = 0.5
lam2 = 0.5
lam_vec = [lam1, lam2, lam2, lam2, lam2] 

S_array = np.insert(S_X_array[t], 0, S_0, axis=0)
Omega_list = [cov.graph_lasso(S_array[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 
A_list = [Omega_list[0]+Omega_list[k+1] for k in range(K)]

log_likelihood = 0
for k in range(K):
    tr_OSOA = 0
    S_Y_kt = S_Y_list[k][t*p+np.array(range(p))[:, None], t*p+np.array(range(p))[None, :]]
    OSOA = np.matmul(np.matmul(np.matmul(Omega_list[k+1], S_Y_kt), Omega_list[k+1]), alg.inv(A_list[k]))
    tr_OSOA = np.trace(OSOA)
    log_det_A = np.log(alg.det(A_list[k]))
    log_det_O = np.log(alg.det(Omega_list[k+1]))
    SO = np.matmul(S_Y_kt, Omega_list[k+1])
    tr_SO = np.trace(SO)
    log_likelihood = log_likelihood + n_vec[k]*(tr_OSOA - log_det_A + log_det_O - tr_SO)

log_likelihood = log_likelihood/n + np.log(alg.det(Omega_list[0]))

penalty = lam_vec[0]*np.sum(np.abs(np.tril(Omega_list[0], -1) + np.triu(Omega_list[0], 1))) 
for k in range(1,K+1):
    penalty = penalty + lam_vec[k]*np.sum(np.abs(np.tril(Omega_list[k], -1) + np.triu(Omega_list[k], 1)))*n_vec[k-1]/n  
pen_likelihood = log_likelihood - penalty  
pen_likelihood0 = 0

while np.abs(pen_likelihood - pen_likelihood0) > 1e-4:       
    # E
    S_0_pd_list1 = []
    S_K_list = []
    for k in range(K):
        S_Y_kt = S_Y_list[k][t*p+np.array(range(p))[:, None], t*p+np.array(range(p))[None, :]]
        OSO = np.matmul(np.matmul(Omega_list[k+1], S_Y_kt), Omega_list[k+1])
        A_inv = alg.inv(A_list[k])
        S_0 = A_inv + np.matmul(np.matmul(A_inv, OSO), A_inv)
        AOS = np.matmul(np.matmul(A_inv, Omega_list[k+1]), S_Y_kt)
        SOA = np.matmul(S_Y_kt, np.matmul(Omega_list[k+1], A_inv))
        S_k = S_Y_kt - AOS - SOA + S_0
        S_0_pd_list1.append(S_0)
        S_K_list.append(S_k)
    S_01 = np.sum(np.array([S_0_pd_list1[k]*n_vec[k]/n for k in range(K)]), axis = 0)
    S_array = np.insert(np.array(S_K_list), 0, S_01, axis=0)
    
    # M
    Omega_list = [cov.graph_lasso(S_array[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 
    A_list = [Omega_list[0]+Omega_list[k+1] for k in range(K)]
    
    log_likelihood = 0
    for k in range(K):
        tr_OSOA = 0
        S_Y_kt = S_Y_list[k][t*p+np.array(range(p))[:, None], t*p+np.array(range(p))[None, :]]
        OSOA = np.matmul(np.matmul(np.matmul(Omega_list[k+1], S_Y_kt), Omega_list[k+1]), alg.inv(A_list[k]))
        tr_OSOA = np.trace(OSOA)
        log_det_A = np.log(alg.det(A_list[k]))
        log_det_O = np.log(alg.det(Omega_list[k+1]))
        SO = np.matmul(S_Y_kt, Omega_list[k+1])
        tr_SO = np.trace(SO)
        log_likelihood = log_likelihood + n_vec[k]*(tr_OSOA - log_det_A + log_det_O - tr_SO)
    
    log_likelihood = log_likelihood/n + np.log(alg.det(Omega_list[0]))
    
    penalty = lam_vec[0]*np.sum(np.abs(np.tril(Omega_list[0], -1) + np.triu(Omega_list[0], 1))) 
    for k in range(1,K+1):
        penalty = penalty + lam_vec[k]*np.sum(np.abs(np.tril(Omega_list[k], -1) + np.triu(Omega_list[k], 1)))*n_vec[k-1]/n  
    pen_likelihood0 = pen_likelihood        
    pen_likelihood = log_likelihood - penalty  
    print(pen_likelihood)

#################################################################################################################
        
        
        
        
    OSO = np.zeros([p,p])
    S_K_list = []
    for m in range(K):
        SO = np.zeros([p,p])
        OS = np.zeros([p,p])
        for l in range(K):
            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
            S_lm = S_Y[l*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
            SO = SO + np.matmul(S_ml, Omega_list[l+1])
            OS = OS + np.matmul(Omega_list[l+1], S_lm)
            OSO = OSO + np.matmul(np.matmul(Omega_list[m+1],S_ml), Omega_list[l+1])
        S_mm = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
        S_k = S_mm - np.matmul(SO, A_inv) - np.matmul(A_inv, OS)
        S_K_list.append(S_k)

    S_0 = A_inv + np.matmul(np.matmul(A_inv,OSO), A_inv)
    S_K_list = [item + S_0 for item in S_K_list]
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
    
    pen_likelihood0 = pen_likelihood        
    pen_likelihood = likelihood - penalty
    print(pen_likelihood)










#############################################################################################################

lam1 = 0.5
lam2 = 0.5
lam_vec = [lam1, lam2, lam2, lam2, lam2] 
beta = 0.5

#initialize
Omega_T_list = [TVGL_xie(S_pd0_T_array[k], lam_vec[k], beta, 3, verbose = True) for k in range(K+1)] 

Omega_T_array = np.array(Omega_T_list)
Omega_T_array = Omega_T_array.transpose([1,0,2,3])

pen_likelihood_T = 0
for time_ix in range(len_t):
    Omega_list = Omega_T_array[time_ix]
    S_Y = S_Y_list[time_ix]
    
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
    print pen_likelihood
    pen_likelihood_T = pen_likelihood_T + pen_likelihood
    print pen_likelihood_T

pen_likelihood0_T = 0
    



###########################



    

lam_max_vec = [alpha_max(item) for item in S_pd0_list]
lam1_max = lam_max_vec[0]
lam2_max = max(lam_max_vec[1:])
lam1_vec = np.logspace(np.log10(lam1_max*5e-1), np.log10(lam1_max), set_length)[::-1]
lam2_vec = np.logspace(np.log10(lam2_max*5e-1), np.log10(lam2_max), set_length)[::-1]

lam_product = itertools.product(lam1_vec, lam2_vec)
lam_product_list = list(lam_product)

Omega_grid_list = []
for i in range(set_length):
    lam1_0 = lam_product_list[set_length*i][0]
    lam2_0 = lam_product_list[set_length*i][1]
    lam_vec = [lam1_0]+list(np.repeat(lam2_0, K))
    
    #initialize
    Omega_list = [cov.graph_lasso(S_pd0_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 
    
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

    Omega_lam1_list = []     
    for j in range(set_length):
        lam2 = lam_product_list[set_length*i+j][1]
        lam_vec = [lam1_0] + list(np.repeat(lam2, K))
        pen_likelihood0 = 0
        while np.abs(pen_likelihood - pen_likelihood0) > 1e-4:       
            OSO = np.zeros([p,p])
            S_K_list = []
            for m in range(K):
                SO = np.zeros([p,p])
                OS = np.zeros([p,p])
                for l in range(K):
                    S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
                    S_lm = S_Y[l*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
                    SO = SO + np.matmul(S_ml, Omega_list[l+1])
                    OS = OS + np.matmul(Omega_list[l+1], S_lm)
                    OSO = OSO + np.matmul(np.matmul(Omega_list[m+1],S_ml), Omega_list[l+1])
                S_mm = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
                S_k = S_mm - np.matmul(SO, A_inv) - np.matmul(A_inv, OS)
                S_K_list.append(S_k)
        
            S_0 = A_inv + np.matmul(np.matmul(A_inv,OSO), A_inv)
            S_K_list = [item + S_0 for item in S_K_list]
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
            
            pen_likelihood0 = pen_likelihood        
            pen_likelihood = likelihood - penalty
            print(pen_likelihood)
        Omega_lam1_list.append(Omega_list)
        print([i, j])
    Omega_grid_list.append(Omega_lam1_list)
return Omega_grid_list