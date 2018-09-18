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
from scipy import io

p = 100
d = 3
n_change = 3
len_t = 101
n = 1
h = 5.848/np.cbrt(len_t)
sigma = 0.5

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
    E = E0 + np.random.multivariate_normal(mean = np.zeros(p), cov = C, size = n)
    E0 = E
    Y = X+E
    Y_list.append(Y)
    
Y_array = np.array(Y_list) # t by n by p
Y_mean = np.mean(Y_array, 0)
E_array = Y_array - Y_mean # t by n by p

Y_array = Y_array.transpose([1,0,2])
E_array = E_array.transpose([1,0,2])

YYT = np.array([[np.matmul(Y_array[i][t][:,None], Y_array[i][t][None,:]) for t in range(len_t)] for i in range(n)])
EET = np.array([[np.matmul(E_array[i][t][:,None], E_array[i][t][None,:]) for t in range(len_t)] for i in range(n)])

YYT0 = YYT[0]
EET0 = EET[0]
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
        w_ml = np.exp(-np.square(m - l)/h)
        print(w_ml)
        S_Y0 = S_Y0 + S_Yl*w_ml
        S_E0 = S_E0 + S_El*w_ml
        w = w + w_ml      
#    S_0 = S_0/((len_t-1)*len_t)
    S_Y0 = S_Y0/w
    S_E0 = S_E0/w
    S_Y0_list.append(S_Y0)
    S_E0_list.append(S_E0)













