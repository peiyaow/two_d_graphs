#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:37:32 2018

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

Y = io.loadmat('fMRI_AD.mat')['data0']
K = Y.shape[1]
p = Y[0,0].shape[1]
len_t = Y[0,0].shape[2]
n_vec = [Y[0,k].shape[0] for k in range(K)]
n = sum(n_vec)
h = 5.848/np.cbrt(len_t)

Y_list = [Y[0,k].transpose([0, 2, 1]).reshape([n_vec[k], len_t*p]) for k in range(K)] # ni by t by p to ni by tp

# select one subject
Y_array = Y_list[2][10]
S_Y = np.outer(Y_array, Y_array)

S_X_cov = np.zeros([p, p]) # Sigma_0
w = 0
for m in range(len_t):
    for l in range(len_t): 
#    for l in [i for i in range(len_t) if i != m]:
        S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
        print S_ml[0][0]
        w_ml = 1-np.exp(-np.square((m - l)/(h*(len_t-1))))
#        print m,l,w_ml
        w = w + w_ml
        S_X_cov = S_X_cov + S_ml*w_ml
S_X_cov = S_X_cov/w
S_X_cov = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(S_X_cov.ravel()), nrow=p)))

