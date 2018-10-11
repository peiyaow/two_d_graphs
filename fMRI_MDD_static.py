import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
#sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
from two_d_graphs.myfunctions import *
from scipy import io
from statsmodels.stats.moment_helpers import cov2corr
import pandas as pd

# lab computer
data = io.loadmat('/home/peiyao/fMRI_data/fMRI_MDD.mat')['data']

# my own laptop
data = io.loadmat('/Users/MonicaW/Documents/Research/graph_matlab/MDD/fMRI_MDD.mat')['data']

data[0,0] = data[0,0][np.array([i for i in range(100) if i != 60]),:,:] # delete No.61
K = data.shape[1]
p = data[0,0].shape[2]
len_t = data[0,0].shape[1]
n_vec = [data[0,k].shape[0] for k in range(K)]

S_list = []
C_list = []
for k in range(K):
    Y = data[0,k]
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

S_MDD_array = np.array(S_list[1])
C_MDD_array = np.array(C_list[1])

# select subset
score_data = pd.read_excel('/home/peiyao/fMRI_data/Alldata_200.xls')
score = score_data['HAMD']
score_MDD = np.array(score[:100])

ix_list = []
for a in np.unique(score_MDD):
    ix = np.argsort(score_MDD)[(score_MDD == a)[np.argsort(score_MDD)]][0]
    ix_list.append(ix)

score = score_MDD[ix_list]
C_MDD_array = C_MDD_array[ix_list]
n = C_MDD_array.shape[0]
alpha_i = [alpha_max(C_MDD_array[i]) for i in range(n)]
C_MDD_array = C_MDD_array[range(1)+range(2,n)]
# select subset 

C_MDD_array = C_MDD_array[range(10)]
C_0 = np.mean(C_MDD_array, 0)
alpha_1 = alpha_max(C_0)
alpha_0 = alpha_1*0.1
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
S_0_list = [cov.graph_lasso(C_0, alpha)[1] for alpha in alphas]
Omega = S_0_list[5]
G0 = nx.from_numpy_array(Omega)
nx.draw(G0)


from TVGL.inferGraphLaplacian import *
from TVGL.TVGL import *
gvx = TGraphVX()
n = C_MDD_array.shape[0]
indexOfPenalty = 3
lamb = alphas[5]
beta = 5
for i in range(n):
    n_id = i
    S = semidefinite(p, name='S')
#    D = Variable(p,p, name='D')
    obj = -log_det(S) + trace(C_MDD_array[i] * S)  # + alpha*norm(S,1)

    gvx.AddNode(n_id, obj)
    if i > 0:  # Add edge to previous timestamp
        prev_Nid = n_id - 1
        currVar = gvx.GetNodeVariables(n_id)
        prevVar = gvx.GetNodeVariables(prev_Nid)
        currScore = score[n_id]
        prevScore = score[prev_Nid]
        edge_obj = beta/np.abs(currScore - prevScore) * norm(currVar['S'] - prevVar['S'], indexOfPenalty)
        gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

    # Add fake nodes, edges
    gvx.AddNode(n_id + n)
    gvx.AddEdge(n_id, n_id + n, Objective=lamb * norm(S-Omega, 1))

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

G0 = nx.from_numpy_array(Omega)
nx.draw(G0)
G1 = nx.from_numpy_array(thetaSet[14])
nx.draw(G1)

np.save('Omega.npy', Omega)
np.save('Omega_i.npy', thetaSet)

robjects.r.matrix(robjects.FloatVector(Omega.ravel()), nrow=p)

plt.draw()
plt.savefig('myfig')
import matplotlib.pyplot as plt
plt.draw()
