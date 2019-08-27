#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:24:35 2018

@author: MonicaW
"""

alphas_sbj = [[alpha_max(C) for C in C_array_list[k]] for k in range(K)]
alphas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.5), 50) for i in range(n_vec[k])]) for k in range(K)]
betas_sbj_mtx_list = [np.vstack([np.logspace(np.log10(alphas_sbj[k][i]), np.log10(alphas_sbj[k][i]*0.1), 50) for i in range(n_vec[k])]) for k in range(K)]

from TVGL.inferGraphLaplacian import *
from TVGL.TVGL import *
gvx = TGraphVX()
n = C_array_list[0].shape[0]
# n = 10
# indexOfPenalty = 3

#beta = alpha_mtx[0,10] # D
#alpha = alpha_mtx[0,20] # Omega

alpha = alphas_sbj_mtx_list[0][:,15] # S
beta = betas_sbj_mtx_list[0][:,45] # D
#beta = 0.12
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
    gvx.AddEdge(n_id, n_id + n, Objective=alpha[i] * norm(S, 1))
    gvx.AddNode(n_id + 2*n)
    gvx.AddEdge(n_id, n_id + 2*n, Objective=beta[i]*norm(S-Omega, 1)) #lamb*norm(S-Omega, 1)  
#    gvx.AddEdge(n_id, n_id + 2*n, Objective=beta*norm(S-Omega, 4)) #lamb*norm(S-Omega, 1)  
    

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

# S
for ix in range(n):
    print ix, sum(sum(np.abs(thetaSet[ix])!=0))
# D   
for ix in range(n):
    print ix, sum(sum(np.abs(thetaSet[ix] - Omega)!=0))
    
for ix in range(n):
    print ix, np.max(np.triu(np.abs(thetaSet[ix] - Omega),1))

for ix in range(n):
    print ix, np.max(np.abs(np.triu(thetaSet[ix],1)))
    
G0 = nx.from_numpy_array(Omega)
nx.draw(G0)

G1 = nx.from_numpy_array(thetaSet[3])
nx.draw(G1)
    

    

# 4   
# S 15    
# D 45
