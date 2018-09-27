import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
#sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
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
h = 5.848/np.cbrt(len_t)

k = 1
Y = data[0,k].transpose([1,0,2])
n = Y.shape[1]
X_hat = Y.mean(0)
C0_hat = np.dot(np.transpose(X_hat - X_hat.mean(0)), X_hat - X_hat.mean(0))/n

# Y
kernel_vec = np.array([np.exp(-np.square((m - 0)/(h*(len_t-1)))) for m in range(len_t)])
kernel_mtx = np.zeros([len_t, len_t])
for t_ix in range(len_t-1):
    kernel_mtx[t_ix, np.arange(t_ix+1, len_t)] = kernel_vec[np.arange(1, len_t-t_ix)]
kernel_mtx = kernel_mtx + np.transpose(kernel_mtx) + np.eye(len_t)

var_E = []
for i in range(n):
    var_i = []
    Y_mean = np.matmul(kernel_mtx, Y[:,i,:]) / np.sum(kernel_mtx, 1)[:, None]
    for t in range(len_t):
        Y_standard = Y[:,i,:] - Y_mean[t, :]
        var_s = []
        for s in range(len_t):
            var_s.append(kernel_mtx[t,s]*np.outer(Y_standard[s,:], Y_standard[s,:]))
        var_i.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))
    var_E.append(var_i)
C_T_hat = np.array(var_E)






alpha_1 = alpha_max(cov2corr(C0_hat))
alpha_0 = alpha_1*0.5
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
A0_hat_list = [cov.graph_lasso(cov2corr(C0_hat), alpha)[1] for alpha in alphas]

G = nx.from_numpy_array(A0_hat_list[20])
nx.draw_spectral(G)


k = 0
Y = data[0,k].transpose([1,0,2])
n = Y.shape[1]
X_hat = Y.mean(0)
C0_hat = np.dot(np.transpose(X_hat - X_hat.mean(0)), X_hat - X_hat.mean(0))/n

alpha_1 = alpha_max(cov2corr(C0_hat))
alpha_0 = alpha_1*0.5
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
A0_hat_list = [cov.graph_lasso(cov2corr(C0_hat), alpha)[1] for alpha in alphas]

G0 = nx.from_numpy_array(A0_hat_list[20])
nx.draw_spectral(G0)
