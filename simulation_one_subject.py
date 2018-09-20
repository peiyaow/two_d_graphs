#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:19:28 2018

@author: MonicaW
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *
from two_d_graphs.MatrixMaxProj import *
from scipy import io
from statsmodels.stats.moment_helpers import cov2corr

p = 50
d = 2
n_change = 3
len_t = 201
n = 1
h = 5.848/np.cbrt(len_t-1)
sigma = 0.1
phi = 0.5

gG = getGraph(p, d)
S, A = gG
A_T = getGraphatT_Shuheng(S, A, n_change)[1]
A_T_list = [lam*A_T+(1-lam)*A for lam in np.linspace(0, 1, len_t)] # Omega
C_T_list = [getCov(item) for item in A_T_list] # Cov: 0+class time

gG0 = getGraph(p, d)
S0, A0 = gG0
C0 = getCov(A0)

X = np.random.multivariate_normal(mean = np.zeros(p), cov = C0, size = n)
E0 = np.random.multivariate_normal(mean = np.zeros(p), cov = C_T_list[0]*sigma**2, size = n)
Y = X+E0
Y_list = [Y]
for time_ix in range(1,len_t):
    C = C_T_list[time_ix]*sigma**2
    E = np.random.multivariate_normal(mean = np.zeros(p), cov = C, size = n)
#    E = phi*E0 + np.random.multivariate_normal(mean = np.zeros(p), cov = C, size = n)
#    E0 = E
    Y = X+E
    Y_list.append(Y)
    
Y_array = np.array(Y_list) # t by n by p
Y_mean = np.mean(Y_array, 0)
E_array = Y_array - Y_mean # t by n by p

Y_array = Y_array.transpose([1,0,2]) # n by t by p
E_array = E_array.transpose([1,0,2])

#####S_X0
YYT = np.array([[np.matmul(Y_array[i][t][:,None], Y_array[i][t][None,:]) for t in range(len_t)] for i in range(n)])
EET = np.array([[np.matmul(E_array[i][t][:,None], E_array[i][t][None,:]) for t in range(len_t)] for i in range(n)])

YYT0 = YYT[0]
EET0 = EET[0]

# S_X0 = nearestPD(np.mean(YYT0 - EET0, 0)) # psd check np.linalg.cholesky
S_X0 = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(np.mean(YYT0 - EET0, 0).ravel()), nrow=p)))

#####S_X_kernel
S_Y0_list = []
S_E0_list = []
for m in range(len_t):
    S_Y0 = np.zeros([p, p]) # Sigma_0
    S_E0 = np.zeros([p, p]) # Sigma_0
    w = 0
    for l in range(len_t):
        S_Yl = YYT0[l]
        S_El = EET0[l]
#            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
        w_ml = np.exp(-np.square((m - l)/(h*(len_t-1))))
#        print(w_ml)
        S_Y0 = S_Y0 + S_Yl*w_ml
        S_E0 = S_E0 + S_El*w_ml
        w = w + w_ml      
#    S_0 = S_0/((len_t-1)*len_t)
    S_Y0 = S_Y0/w
    S_E0 = S_E0/w
    S_Y0_list.append(S_Y0)
    S_E0_list.append(S_E0)
S_Y0_array = np.array(S_Y0_list)
S_E0_array = np.array(S_E0_list)

# S_X_kernel = nearestPD(np.mean(S_Y0_array - S_E0_array, 0)) # psd check np.linalg.cholesky
S_X_kernel = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(np.mean(S_Y0_array - S_E0_array, 0).ravel()), nrow=p)))
alpha_1 = alpha_max(S_X_kernel)
r_S_X_kernel = robjects.r.matrix(robjects.FloatVector(S_X_kernel.ravel()), nrow=p)
r_path_vec = robjects.FloatVector(np.logspace(np.log10(1), np.log10(0.01), 50))
r_rho_mtx = robjects.r.matrix(robjects.FloatVector(((np.triu(np.ones((p,p)), 1) + np.tril(np.ones((p,p)), -1))*alpha_1).ravel()), nrow = p)
result_S_X_kernel = r_QUIC(r_S_X_kernel, r_rho_mtx, r_path_vec)
Omega_kernel = np.array(result_S_X_kernel[0])

getF1(A0, Omega_kernel[:,:,10])

#####S_X_cov
Y_array = Y_array.reshape([n, len_t*p])
vec_Y_outer = [np.outer(Y_array[i], Y_array[i]) for i in range(n)]

S_Y = vec_Y_outer[0]
S_X_cov = np.zeros([p, p]) # Sigma_0
w = 0
for m in range(len_t):
    for l in range(len_t): 
#    for l in [i for i in range(len_t) if i != m]:
        S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
        w_ml = 1-np.exp(-np.square((m - l)/(h*(len_t-1))))
#        print m,l,w_ml
        w = w + w_ml
        S_X_cov = S_X_cov + S_ml*w_ml
S_X_cov = S_X_cov/w
S_X_cov = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(S_X_cov.ravel()), nrow=p)))

# shrunk covariance
max(np.linalg.eig(S_X_cov)[0])
min(np.linalg.eig(S_X_cov)[0])
S_X_cov_shrunk = cov.shrunk_covariance(S_X_cov, 0.7)
max(np.linalg.eig(S_X_cov_shrunk)[0])
min(np.linalg.eig(S_X_cov_shrunk)[0])
alpha_1 = alpha_max(S_X_cov_shrunk)
alpha_0 = alpha_1*0.01
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
Omega_cov_shrunk = [cov.graph_lasso(S_X_cov_shrunk, alpha)[1] for alpha in alphas]
getF1(A0, Omega_cov_shrunk[19])


# quic
alpha_1 = alpha_max(S_X_cov)
r_S_X_cov = robjects.r.matrix(robjects.FloatVector(S_X_cov.ravel()), nrow=p)
r_path_vec = robjects.FloatVector(np.logspace(np.log10(1), np.log10(0.01), 50))
r_rho_mtx = robjects.r.matrix(robjects.FloatVector(((np.triu(np.ones((p,p)), 1) + np.tril(np.ones((p,p)), -1))*alpha_1).ravel()), nrow = p)
result_S_X_cov = r_QUIC(r_S_X_cov, r_rho_mtx, r_path_vec)
Omega_cov = np.array(result_S_X_cov[0])

getF1(A0, Omega_cov[:,:,30])

