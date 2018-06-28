#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:37:25 2018

@author: MonicaW
"""

np.random.seed(10)
p = 5
d = 1
#Rnd = snap.TRnd(10)
UGraph = snap.GenPrefAttach(p, d)
for EI in UGraph.Edges():
# generate a random number in (-1, -0.5)U(0.5,1)
#    r = np.random.uniform(0, 1.)
#    print r
    print EI.GetSrcNId(), EI.GetDstNId()
    
np.random.seed(10)
random.seed(10)
p = 5
d = 1
G6 = snap.GenRndGnm(snap.PUNGraph, p, int((p*(p-1))*0.05))
# S = np.zeros((size,size))
for EI in G6.Edges():
    print EI.GetSrcNId(), EI.GetDstNId()
    
    
import random
import networkx as nx
import numpy as np
from scipy import sparse
random.seed(10)
np.random.seed(123)
p = 5
d = 1
# G = nx.scale_free_graph(p)
S = nx.barabasi_albert_graph(p, d)  
S = nx.adjacency_matrix(S)
S = sparse.triu(S)
row_ix, col_ix = sparse.find(S)[0:2]
n_nonzero = len(sparse.find(S)[2])
S = S.todense().astype(float)
S0 = S.copy()
for i in range(n_nonzero):
    r = np.random.uniform(0, 1.)
    S[row_ix[i], col_ix[i]] = r-1. if r < 0.5 else r

vec_div = 1.5*np.sum(np.absolute(S), axis = 1) 
for i in range(p):
    if vec_div[i]: 
        # only when the absolute value of the vector is not zero do the standardization
        S[i,:] = S[i,:]/vec_div[i]
A = (S + S.T)/2 + np.matrix(np.eye(p))
# check if A is PD
print(np.all(alg.eigvals(A) > 0))









Rnd = snap.TRnd()
UGraph = snap.GenPrefAttach(p, d, Rnd)
S = np.zeros((p,p))
for EI in UGraph.Edges():
# generate a random number in (-1, -0.5)U(0.5,1)
# method: https://stats.stackexchange.com/questions/270856/how-to-randomly-generate-random-numbers-in-one-of-two-intervals
    r = np.random.uniform(0, 1.)
    print r
    print EI.GetSrcNId(), EI.GetDstNId()
    S[EI.GetSrcNId(), EI.GetDstNId()] = r-1. if r < 0.5 else r  # assign values to edges

S0 = S.copy() 
# orginal half graph without standardizing it into a PD matrix
#    S =  S + S.T
# the whole row could be all zeros
vec_div = 1.5*np.sum(np.absolute(S), axis = 1)[:,None] 






