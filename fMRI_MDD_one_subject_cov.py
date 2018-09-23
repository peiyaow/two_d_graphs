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
ix = 3
Y = data[0,1][ix]
Y_stack = Y.reshape([1, len_t*p])
S_Y = np.outer(Y_stack, Y_stack)

S_X_cov = np.zeros([p, p]) # Sigma_0
w = 0
for m in range(len_t):
    for l in range(len_t):
        S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
        w_ml = 1-np.exp(-np.square((m - l)/(h*(len_t-1))))
        w = w + w_ml
        S_X_cov = S_X_cov + S_ml*w_ml
S_X_cov = S_X_cov/w

