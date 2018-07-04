#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:19:52 2018

@author: monicawang76
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

p0 = 20
p1 = 20
d0 = 1
d1 = 1
ni = 50
n_change = 2
len_t = 11

A_list, C_list, X_list = simulate_data(p0, p1, d0, d1, ni, n_change, len_t)

len_class = len(X_list)
set_length = 51
p = X_list[0][0].shape[1]

product = itertools.product(range(len_class), range(len_t))
mesh_product = list(product)
#-------------------------------------------------------------------------------------------------------------------------------

pool = mp.Pool(processes=10)
Theta_glasso_list = [pool.apply(myglasso, args=(X_list, ix_product, set_length)) for ix_product in mesh_product]

Theta_glasso_array = np.array(Theta_glasso_list) # 44 class by time
Theta_glasso_array = np.reshape(Theta_glasso_array, (len_class, len_t, set_length, p, p))

filename = 'glasso' + str(sim_ix) + '.npy' 
np.save(filename, Theta_glasso_array)










