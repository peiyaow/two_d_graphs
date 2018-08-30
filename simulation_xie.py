# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append('/nas/longleaf/home/peiyao/proj2')

import numpy as np
from two_d_graphs.myfunctions import *
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
ni = 50
n_change = 3
len_t = 50
K = 4
h = 5.848/np.cbrt(len_t)
width = 5
set_length = 50 # number of lambdas

A_list, O_list, C_list, Y_array = simulate_data_xie() # Y_array: time class n p
Y_array = np.transpose(Y_array, [0, 2, 1, 3]) # time n class p
Y_array = np.reshape(Y_array, [len_t, n_vec[0], K*p]) # time n Kp

S_Y_list = [genEmpCov(Y_array[t].T, useKnownMean = True) for t in range(len_t)]

pool = mp.Pool(processes=10)
Omega_list = [pool.apply(EM_xie_time_varying, args=(S_Y, p, K, set_length)) for S_Y in S_Y_list]
            
Omega_array = np.array(Omega_grid_list) # time lam1 lam2 K p p
Omega_array = Omega.array.transpose([3, 0, 1, 2, 4, 5]) # K time lam1 lam2 p p
Omega_array = Omega.array.reshape([K, len_t, set_length**2, p, p]) # K time lam1*lam2 p p

A_array = np.array(A_list) # K time p p
# A_array = A_array.transpose([1, 0, 2, 3]) # time K p p

PD_result = []
for k in range(K+1):
    PD_result.append(PD_array_simple(Omega_array, A_array, k, 25))


        
        
        
        
        
        
        
        
        
 