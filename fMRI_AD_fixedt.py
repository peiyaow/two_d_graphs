#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:59:49 2018

@author: MonicaW
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *

from scipy import io

Y = io.loadmat('fMRI_AD.mat')['data']
K = Y.shape[1]
p = Y[0,0].shape[1]
len_t = Y[0,0].shape[2]
n_vec = [Y[0,k].shape[0] for k in range(K)]
n = sum(n_vec)
h = 5.848/np.cbrt(len_t)

Y_list = [Y[0,k].transpose([0,2,1]).reshape([n_vec[k], p*len_t]) for k in range(K)] # ni by p by t -> each element: ni by time*p

S_Y_list = [np.matmul(Y_list[k].T, Y_list[k])/n_vec[k] for k in range(K)]

S_0_list = []
S_0_pd_list = []
for k in range(K): 
    S_Y = S_Y_list[k]
    S_0 = np.zeros([p, p]) # Sigma_0
    w = 0
    for m in range(len_t):
        for l in [i for i in range(len_t) if i != m]:
            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
            w_ml = np.exp(-np.square(m - l)/h)
#            print(w_ml)
            w = w + w_ml
            S_0 = S_0 + S_ml*w_ml
#    S_0 = S_0/((len_t-1)*len_t)
    S_0 = S_0/w
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
t = 0
S_array = np.insert(S_X_array[t], 0, S_0, axis=0)
S_array0 = S_array
alpha_max(S_array0[2])

lam1 = 5
lam2 = 30
lam_vec = [lam1, lam2, lam2, lam2, lam2] 

np.logspace(np.log10(80*5e-2), np.log10(80), 2)

Omega_list = [cov.graph_lasso(S_array[k], lam_vec[k], verbose = False, max_iter=5000)[1] for k in range(K+1)] 
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
    print alpha_max(S_array[0])
    print alpha_max(S_array[1])

