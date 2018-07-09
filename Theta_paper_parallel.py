#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:27:42 2018

@author: monicawang76
"""

import sys
sys.path.append('/nas/longleaf/home/peiyao/proj2')
from two_d_graphs.myfunctions import *
import TVGL.TVGL as tvgl
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
indexOfPenalty = 1

alpha_upper = alpha_max(genEmpCov(X_list[3][10].T))
alpha_lower = alpha_max(genEmpCov(X_list[0][0].T))

alpha_set = np.logspace(np.log10(alpha_lower*5e-2), np.log10(alpha_upper), set_length_alpha)
beta_set = np.logspace(-2, 0.5, set_length_beta)
parameters_product = itertools.product(alpha_set, beta_set)
mesh_parameters = list(parameters_product)
#-------------------------------------------------------------------------------------------------------------------------------
X_concat_list = []
for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    X_concat_list.append(X_concat)

pool = NoDaemonProcessPool(processes=10)
results_paper_list = []
for class_ix in range(len_class):
    results_paper = [pool.apply(tvgl.TVGL, args=(X_concat_list[class_ix], ni, parameters[0], parameters[1], indexOfPenalty)) for parameters in mesh_parameters]
    results_paper_list.append(results_paper)

tvgl.TVGL(X_concat_list[0], ni, 0.5, 1, indexOfPenalty)

Theta_paper_array = np.array(results_paper_list) # class by alpha by beta by t by p by p 
Theta_paper_array = np.reshape(Theta_paper_array, (len_class, set_length_alpha, set_length_beta, len_t, p, p)) #  class, alpha, beta, time, p, p
Theta_paper_array = np.transpose(Theta_paper_array, [2, 0, 3, 1, 4, 5]) # beta, class, time, alpha, row, col

filename = 'paper' + str(sim_ix) + '.npy' 
np.save(filename, Theta_paper_array)























