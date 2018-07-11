#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:27:42 2018

@author: monicawang76
"""

import sys
sys.path.append('/nas/longleaf/home/peiyao/proj2')
from two_d_graphs.myfunctions import *
import TVGL.myTVGL as mtvgl
import numpy as np
import random
import itertools

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

set_length_alpha = 51
set_length_beta = 51
indexOfPenalty = 3

alpha_upper = alpha_max(genEmpCov(X_list[3][10].T))
alpha_lower = alpha_max(genEmpCov(X_list[0][0].T))

alpha_set = np.logspace(np.log10(alpha_lower*5e-2), np.log10(alpha_upper), set_length_alpha)
beta_set = np.logspace(-2, 0.5, set_length_beta)
parameters_product = itertools.product(alpha_set, beta_set)
mesh_parameters = list(parameters_product)

h = 5.848/np.cbrt(ni*len_t)
#-------------------------------------------------------------------------------------------------------------------------------

X_array = np.array(X_list) # class by time by ni by p
X_array = np.reshape(X_array, (len_class, len_t*ni, p)) # class by ni*len_t by p
    
pool = NoDaemonProcessPool(processes=10)
results_mymethod = [pool.apply(mtvgl.myTVGL, args=(X_array, ni, parameters[0], parameters[1], indexOfPenalty, True, h)) for parameters in mesh_parameters]

# mtvgl.myTVGL(X_array, ni, 0.5, 1, indexOfPenalty, True, h)

Theta_mymethod_array = np.array(results_mymethod) # alpha*beta, time, class, p, p
Theta_mymethod_array = np.reshape(Theta_mymethod_array, (set_length_alpha, set_length_beta, len_t, len_class, p, p)) # alpha, beta, time, class, p, p
Theta_mymethod_array = np.transpose(Theta_mymethod_array, [1, 3, 2, 0, 4, 5]) # beta, class, time, alpha, row, col

class_ix = 3
PD_result = PD_array(Theta_mymethod_array, A_list, class_ix)

filename = 'mymethod' + str(sim_ix) + '.npy' 
np.save(filename, PD_result)
