#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:42:30 2018

@author: MonicaW
"""

import sys
sys.path.append('/nas/longleaf/home/peiyao/proj2')
from two_d_graphs.myfunctions import *
import numpy as np
import random
import itertools
import multiprocessing as mp

#------------------------------------------- retrieve data ---------------------------------------------------------------------
myseed1 = int(sys.argv[1])
myseed2 = int(sys.argv[2])
sim_ix = int(sys.argv[3])

random.seed(myseed1)
np.random.seed(myseed2)

p = 100
d = 3
n_vec = [50, 50, 50, 50]
ni = 50
n_change = 3
len_t = 50
K = 4
h = 5.848/np.cbrt(len_t)
width = 5
set_length = 51 # number of lambdas

A_list, O_list, C_list, Y_array = simulate_data_xie() # Y_array: time class n p
Y_array = np.transpose(Y_array, [1, 0, 2, 3]) # class time n p

product = itertools.product(range(K), range(len_t))
mesh_product = list(product)
#-------------------------------------------------------------------------------------------------------------------------------
pool = mp.Pool(processes=10)
Theta_Shuheng_list = [pool.apply(Shuheng_method, args=(Y_array, ix_product, set_length, h, width, True)) for ix_product in mesh_product]

Theta_Shuheng_array = np.array(Theta_Shuheng_list) 
Theta_Shuheng_array = np.reshape(Theta_Shuheng_array, (K, len_t, set_length, p, p)) # class by time by alpha by p by p

PD_result = []
for k in range(K):
    PD_result.append(PD_array_simple(Theta_Shuheng_array, O_list, k, 25))

PD_result = np.array(PD_result)
filename = 'Shuheng' + str(sim_ix) + '.npy' 
np.save(filename, PD_result)
