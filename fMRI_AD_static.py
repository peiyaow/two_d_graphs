import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
#sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
from two_d_graphs.myfunctions import *
from scipy import io
from statsmodels.stats.moment_helpers import cov2corr
import pandas as pd

# lab
data = io.loadmat('/home/peiyao/fMRI_data/fMRI_AD.mat')['data0']

# laptop
data = io.loadmat('/Users/MonicaW/Documents/Research/graph_matlab/ADNI/fMRI_AD.mat')['data0']
K = data.shape[1]
p = data[0,0].shape[1]
len_t = data[0,0].shape[2]
n_vec = [data[0,k].shape[0] for k in range(K)]
n = sum(n_vec)
# h = 5.848/np.cbrt(len_t)
# data = [data[0,k].transpose([0, 2, 1]) for k in range(K)] # ni by t by p
data = [data[0,k] for k in range(K)] # ni by t by p

S_list = []
C_list = []
for k in range(K):
    Y = data[k]
    n = Y.shape[0] # number of subjects
    S_k_list = []
    C_k_list = []
    for i in range(n):
        S = np.dot(np.transpose(Y[i]), Y[i])/len_t
        C = cov2corr(S)
        S_k_list.append(S)
        C_k_list.append(C)
    S_list.append(S_k_list)
    C_list.append(C_k_list)

C_array_list = [np.array(C_list[k][:10]) for k in range(K)]
C_mean_array_list = [C_array_list[k].mean(0) for k in range(K)]

# alpha_mtx for mean
alphas = [alpha_max(C_mean_array_list[k]) for k in range(K)]
alpha_mtx = np.vstack([np.logspace(np.log10(np.array(alphas))[k], np.log10(np.array(alphas)*0.1)[k], 50) for k in range(K)])

# alpha_mtx for each subject
alphas_sbj = [[alpha_max(C) for C in C_array_list[k]] for k in range(K)]
# alphas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.5), 50) for i in range(10)]) for k in range(K)]
alphas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.5), 50) for i in range(n_vec[k])]) for k in range(K)]

S_0_list = [[cov.graph_lasso(C_mean_array_list[k], alpha)[1] for alpha in alpha_mtx[k]] for k in range(K)]
S_0_array = np.array(S_0_list).transpose([1,0,2,3])

Omega = S_0_array[5][0]
sum(sum(np.abs(Omega)!=0))

from TVGL.inferGraphLaplacian import *
from TVGL.TVGL import *
gvx = TGraphVX()
n = C_array_list[0].shape[0]
indexOfPenalty = 3
beta = alpha_mtx[0,10] # D
alpha = alpha_mtx[0,20] # Omega
for i in range(n):
    n_id = i
    S = semidefinite(p, name='S')
#    D = Variable(p,p, name='D')
    obj = -log_det(S) + trace(C_array_list[0][i] * S)  # + alpha*norm(S,1)
    gvx.AddNode(n_id, obj)

    # if i > 0:  # Add edge to previous timestamp
    #     prev_Nid = n_id - 1
    #     currVar = gvx.GetNodeVariables(n_id)
    #     prevVar = gvx.GetNodeVariables(prev_Nid)
    #     currScore = score[n_id]
    #     prevScore = score[prev_Nid]
    #     edge_obj = beta/np.abs(currScore - prevScore) * norm(currVar['S'] - prevVar['S'], indexOfPenalty)
    #     gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

    # Add fake nodes, edges
    gvx.AddNode(n_id + n)
    gvx.AddEdge(n_id, n_id + n, Objective=beta*norm(S-Omega, 1)) #lamb*norm(S-Omega, 1)
    gvx.AddNode(n_id + 2*n)
    gvx.AddEdge(n_id, n_id + 2*n, Objective=alpha * norm(S, 1))

verbose = True
epsAbs = 1e-3
epsRel = 1e-3
eps = 3e-3

gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose=verbose)

thetaSet = []
for nodeID in range(n):
    val = gvx.GetNodeValue(nodeID, 'S')
    thetaEst = upper2FullTVGL(val, eps)
    thetaSet.append(thetaEst)

for ix in range(n):
    print ix, sum(sum(np.abs(thetaSet[ix])!=0))
