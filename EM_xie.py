# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:42:53 2018

@author: peiyao
"""

import numpy as np
import numpy.linalg as alg
import sklearn.covariance as cov

from myfunctions import *
from posdef import *

lam1 = 0.5
lam2 = 0.5
lam_vec = [lam1, lam2, lam2, lam2, lam2] 

#initialize
Omega_list = [cov.graph_lasso(S_pd_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 

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
pen_likelihood0 = 0

while np.abs(pen_likelihood - pen_likelihood0) > 1e-4:
    #E
#    OSO = np.zeros([p,p])
#    for m in range(K):
#        for l in range(K):
#            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
#            OSO = OSO + np.matmul(np.matmul(Omega_list[m+1],S_ml), Omega_list[l+1])
#    
#    S_0 = A_inv + np.matmul(np.matmul(A_inv,OSO), A_inv)      
#    
#    S_K_list = []
#    for m in range(K):
#        SO = np.zeros([p,p])
#        OS = np.zeros([p,p])
#        for l in range(K):
#            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
#            S_lm = S_Y[l*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
#            SO = SO + np.matmul(S_ml, Omega_list[l+1])
#            OS = OS + np.matmul(Omega_list[l+1], S_lm)
#        S_k = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]] - np.matmul(SO, A_inv) - np.matmul(A_inv, OS) + S_0
#        S_K_list.append(S_k)
#    S_pd_list = [S_0] + S_K_list
    
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
      
# np.diag(S_pd_list[0])