import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
#sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
from two_d_graphs.myfunctions import *
from scipy import io
from statsmodels.stats.moment_helpers import cov2corr
import pandas as pd

# my laptop
data = io.loadmat('/Users/MonicaW/Documents/Research/graph_matlab/ADNI/AD_array.mat')['timeseries_AD']
label = np.loadtxt('/Users/MonicaW/Documents/Research/graph_matlab/ADNI/label.txt')
ix = np.loadtxt('/Users/MonicaW/Documents/Research/graph_matlab/ADNI/ix.txt')

n = data.shape[0]
len_t = data.shape[1]
ix = ix-1
label = label-1
ix = ix.astype(int)
label = label.astype(int)
N = np.unique(ix).shape[0]
K = np.unique(label).shape[0]

C_list = []
for i in range(n):
    S = np.dot(np.transpose(data[i,:,:]), data[i,:,:])/len_t
    C = cov2corr(S)
    C_list.append(C)

C_array = np.array(C_list)
C_subject_list = []
for i in range(N):
    n_i = sum(ix==i)
#     print n_i
    C = C_array[ix==i,:,:].sum(0)/n_i
    C_subject_list.append(C)
C_subject_array = np.array(C_subject_list)

C_list = [C_subject_array[np.unique(ix[label==k]),:,:] for k in range(K)]

C_array_list = [np.array(C_list[k]) for k in range(K)]
C_mean_array_list = [C_array_list[k].mean(0) for k in range(K)]

n_vec = [C_array_list[k].shape[0] for k in range(K)]

alphas = [alpha_max(C_mean_array_list[k]) for k in range(K)]
alpha_mtx = np.vstack([np.logspace(np.log10(np.array(alphas))[k], np.log10(np.array(alphas)*0.5)[k], 50) for k in range(K)])

# alpha_mtx for each subject
alphas_sbj = [[alpha_max(C) for C in C_array_list[k]] for k in range(K)]
# alphas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.5), 50) for i in range(10)]) for k in range(K)]
alphas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.5), 50) for i in range(n_vec[k])]) for k in range(K)]

S_0_list = [[cov.graph_lasso(C_mean_array_list[k], alpha)[1] for alpha in alpha_mtx[k]] for k in range(K)]
S_0_array = np.array(S_0_list).transpose([1,0,2,3])

Omega = S_0_array[5][0]
sum(sum(np.abs(Omega)!=0))









