#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:27:42 2018

@author: monicawang76
"""

import sys
sys.path.append('/nas/longleaf/home/peiyao/proj2')
sys.path.append('..')
from two_d_graphs.myfunctions import *
import TVGL.myTVGL as mtvgl
import numpy as np
import random
import itertools

#------------------------------------------- retrieve data ---------------------------------------------------------------------
myseed1 = int(sys.argv[1])
myseed2 = int(sys.argv[2])
sim_ix = int(sys.argv[3])

myseed1 = int(706)
myseed2 = int(131)
sim_ix = int(1)

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

alpha_set = np.logspace(np.log10(0.9*5e-2), np.log10(0.9), set_length_alpha)
beta_set = np.logspace(-2, 0.5, set_length_beta)
parameters_product = itertools.product(alpha_set, beta_set)
mesh_parameters = list(parameters_product)

h = 5.848/np.cbrt(ni*len_t)
#-------------------------------------------------------------------------------------------------------------------------------

X_array = np.array(X_list)
X_array = np.transpose(X_array, [1, 0, 2, 3]) # time by class by ni by p
X_array = np.reshape(X_array, (len_t, len_class*ni, p)) # time by ni*len_class by p
result = mtvgl.myTVGL0(X_array, ni, 0.1, 0.1, 1, True, h)


X_array = np.array(X_list) # class by time by ni by p
X_array = np.reshape(X_array, (len_class, len_t*ni, p)) # time by ni*len_class by p
result = mtvgl.myTVGL(X_array, ni, 0.1, 1, 1, True, h)
#X_concat_class_list = []
#for class_ix in range(len_class):
#    X_concat = np.concatenate(X_list[class_ix])
#    X_concat_class_list.append(X_concat)

pool = NoDaemonProcessPool(processes=10)
results_paper_list = []
for class_ix in range(len_class):
    results_paper = [pool.apply(mtvgl.TVGL, args=(X_concat_class_list[class_ix], ni, parameters[0], parameters[1], indexOfPenalty, True, h)) for parameters in mesh_parameters]
    results_paper_list.append(results_paper)

result = [pool.apply(mtvgl.myTVGL0, args=(X_array, ni, parameters[0], parameters[1], indexOfPenalty, True, h)) for parameters in mesh_parameters]


Theta_paper_array = np.array(results_paper_list) # alpha by beta
Theta_paper_array = np.reshape(Theta_paper_array, (set_length_alpha, set_length_beta, len_t, len_class, p, p)) # alpha beta t class p p
Theta_paper_array = np.transpose(Theta_paper_array, [1, 3, 2, 0, 4, 5]) # beta, class, time, alpha, row, col

filename = 'paper' + str(sim_ix) + '.npy' 
np.save(filename, Theta_paper_array)



















