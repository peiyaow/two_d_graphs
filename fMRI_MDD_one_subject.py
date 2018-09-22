#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:26:38 2018

@author: peiyao
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

# lab computer
data = io.loadmat('/home/peiyao/fMRI_data/fMRI_MDD.mat')['data']

# my own laptop
data = io.loadmat('/Users/MonicaW/Documents/Research/graph_matlab/MDD/fMRI_MDD.mat')['data']

data[0,0] = data[0,0][np.array([i for i in range(100) if i != 60]),:,:] # delete No.61
K = data.shape[1]
p = data[0,0].shape[2]
len_t = data[0,0].shape[1]
n_vec = [data[0,k].shape[0] for k in range(K)]
n = sum(n_vec)
h = 5.848/np.cbrt(len_t)
len_t_z = len_t/2
# h_z = 5.848/np.cbrt(len_t_z)

# Y
kernel_vec = np.array([np.exp(-np.square((m - 0)/(h*(len_t-1)))) for m in range(len_t)])
#[m*1.0/(len_t-1) for m in range(len_t)]

kernel_mtx = np.zeros([len_t, len_t])
for t_ix in range(len_t-1):
    kernel_mtx[t_ix, np.arange(t_ix+1, len_t)] = kernel_vec[np.arange(1, len_t-t_ix)]  
kernel_mtx = kernel_mtx + np.transpose(kernel_mtx) + np.eye(len_t)

# Z
# kernel_vec_z = np.hstack([np.array([np.exp(-np.square((m - 0)/(h_z*(len_t_z-1)))) for m in range(len_t_z)]), np.zeros([1,len_t-len_t_z])[0,:]])
kernel_vec_z = np.hstack([np.array([np.exp(-np.square((m - 0)/(h*(len_t-1)))) for m in range(len_t_z)]), np.zeros([1,len_t-len_t_z])[0,:]])

#[m*1.0/(len_t-1) for m in range(len_t)]

kernel_mtx_z = np.zeros([len_t, len_t])
for t_ix in range(len_t-1):
    kernel_mtx_z[t_ix, np.arange(t_ix+1, len_t)] = kernel_vec_z[np.arange(1, len_t-t_ix)]  
kernel_mtx_z = kernel_mtx_z + np.transpose(kernel_mtx_z) + np.eye(len_t)

# select one MDD subject
ix = 0
Y = data[0,1][ix]
X = np.mean(Y, 0)
Z = Y - X

#kernel_mtx = kernel_mtx_z
# Z

#Z_mean = np.matmul(kernel_mtx, Z)/np.sum(kernel_mtx, 1)[:,None]
#
#t = 0 # var(z_t)
#s = 0 # kernel k(t,s)
#var_Z = []
#for t in range(len_t):
#    Z_standard = Z - Z_mean[t,:]
#    var_s = []
#    for s in range(len_t):
#        var_s.append(kernel_mtx[t,s]*np.outer(Z_standard[s,:], Z_standard[s,:]))
#    var_Z.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))

Z_mean = np.matmul(kernel_mtx_z, Z)/np.sum(kernel_mtx_z, 1)[:,None]

t = 0 # var(z_t)
s = 0 # kernel k(t,s)
var_Z = []
for t in range(len_t):
    Z_standard = Z - Z_mean[t,:]
    var_s = []
    for s in range(len_t):
        var_s.append(kernel_mtx_z[t,s]*np.outer(Z_standard[s,:], Z_standard[s,:]))
    var_Z.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx_z[t,:]))
    
#t = 0 # var(z_t)
#s = 0 # kernel k(t,s)
#var_Z = []
#for t in range(len_t):
#    var_s = []
#    for s in range(len_t):
#        var_s.append(kernel_mtx[t,s]*np.outer(Z[s,:], Z[s,:]))
#    var_Z.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))

# Y
Y_mean = np.matmul(kernel_mtx, Y)/np.sum(kernel_mtx, 1)[:,None]

t = 0 # var(z_t)
s = 0 # kernel k(t,s)
var_Y = []
for t in range(len_t):
    Y_standard = Y - Y_mean[t,:]
    var_s = []
    for s in range(len_t):
        var_s.append(kernel_mtx[t,s]*np.outer(Y_standard[s,:], Y_standard[s,:]))
    var_Y.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))
    
var_Y_array = np.array(var_Y)

import scipy.io as sio
sio.savemat('var_Y_array.mat', {'var_Y_array':var_Y_array, })

var_Y_mean = np.mean(var_Y_array, 0)

S0 = var_Y_array[0]
S1 = var_Y_array[160]

S0_pd = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(S0.ravel()), nrow=p)))
S1_pd = np.array(r_MatrixMaxProj(robjects.r.matrix(robjects.FloatVector(S1.ravel()), nrow=p)))


test0 = np.linalg.eig(S0_pd)
test1 = np.linalg.eig(S1_pd)

test0 = np.linalg.eig(var_Y_array[0])
test1 = np.linalg.eig(var_Y_array[160])
eigen_mtx = np.vstack([np.real(test0[0]), np.real(test1[0])])
eigen_common = np.min(eigen_mtx,0)
V = (test0[1] + test1[1])/2
V0 = test0[1]
V1 = test1[1]

S0_new = np.dot(np.dot(V, np.diag(test0[0])), np.transpose(V))
S1_new = np.dot(np.dot(V, np.diag(test1[0])), np.transpose(V))

S0_new1 = np.dot(np.dot(V0, np.diag(test0[0])), np.transpose(V0))
S1_new1 = np.dot(np.dot(V1, np.diag(test1[0])), np.transpose(V1))

S = np.dot(np.dot(V, np.diag(eigen_common)), np.transpose(V))

test0 = np.linalg.eig(S0)
test1 = np.linalg.eig(var_Y_array[160])







