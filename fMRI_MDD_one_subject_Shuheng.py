import numpy as np
import sys
import os
sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *
from two_d_graphs.MatrixMaxProj import *
from scipy import io

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

kernel_vec = np.array([np.exp(-np.square((m - 0)/(h*(len_t-1)))) for m in range(len_t)])
kernel_mtx = np.zeros([len_t, len_t])
for t_ix in range(len_t-1):
    kernel_mtx[t_ix, np.arange(t_ix+1, len_t)] = kernel_vec[np.arange(1, len_t-t_ix)]
kernel_mtx = kernel_mtx + np.transpose(kernel_mtx) + np.eye(len_t)

# select one MDD subject
ix = 1
Y = data[0,1][ix]
X = np.mean(Y, 0)
Z = Y - X

var_Z = []
for t in range(len_t): # var(z_t)
    var_s = []
    for s in range(len_t): # kernel k(t,s)
        var_s.append(kernel_mtx[t,s]*np.outer(Z[s,:], Z[s,:]))
    var_Z.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))

var_Y = []
for t in range(len_t):
    var_s = []
    for s in range(len_t):
        var_s.append(kernel_mtx[t,s]*np.outer(Y[s,:], Y[s,:]))
    var_Y.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))

var_Y[0] - var_Z[0]

var_Y_array = np.array(var_Y)
io.savemat('var_Y_array_Shuheng.mat', {'var_Y_array':var_Y_array})
