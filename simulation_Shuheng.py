#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# generate time-varying graph using the idea from Shuheng's Paper
# https://arxiv.org/pdf/0802.2758.pdf

import sys
import os
sys.path.append(os.path.abspath('..')+'/TVGL')
import pickle

import numpy as np
# from random import *
import random
import snap
import numpy.linalg as alg
from scipy.linalg import block_diag
import sklearn.covariance as cov
import TVGL as tvgl
import myTVGL as mtvgl


#---------------------------------------------- Define private functions -----------------------------------------------------
def getGraph(p,d):
    # construct a graph from scale free distribution
    # paper: The joint graphical lasso for inverse covarianceestimation across multiple classes
    # reference: https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/rssb.12033
    # p: number of nodes
    # d: out degree of each node
    Rnd = snap.TRnd(10)
    UGraph = snap.GenPrefAttach(p, d, Rnd)
    S = np.zeros((p,p))
    for EI in UGraph.Edges():
    # generate a random number in (-0.4, -0.1)U(0.1,0.4)
    # method: https://stats.stackexchange.com/questions/270856/how-to-randomly-generate-random-numbers-in-one-of-two-intervals
        r = np.random.uniform(0, 0.6)
        S[EI.GetSrcNId(), EI.GetDstNId()] = r-0.4 if r < 0.3 else r-0.2 # assign values to edges
    
    S0 = S.copy() 
    # orginal half graph without standardizing it into a PD matrix
    S =  S + S.T 
    S = S/(1.5*np.sum(np.absolute(S), axis = 1)[:,None])
    A = (S + S.T)/2 + np.matrix(np.eye(p))
    # check if A is PD
    print(np.all(alg.eigvals(A) > 0))
    return S0, A
#    return A

def getGraphatT_Shuheng(S, A, n_change):
    # n_change: number of edges to be changed
    # n_change edges to be added 
    # n_change edges to be dropped
    p = A.shape[0]
    
    B = S + np.tril(np.ones((p,p))) 
    
    S_where_is_not_zero = np.where(S!=0)
    S_where_is_zero = np.where(B==0)
    
    n_offdiag_nonzero = S_where_is_not_zero[0].shape[0]
    n_offdiag_zero = S_where_is_zero[0].shape[0]
    
    drop_ix = random.sample(range(n_offdiag_nonzero), n_change)
    add_ix = random.sample(range(n_offdiag_zero), n_change) 
    
    w_add = []
    for i in range(n_change):
        r = np.random.uniform(0, 0.6)
        w_add.append(r-0.4 if r < 0.3 else r-0.2)
    
    S_add = np.zeros((p,p))
    S_add[S_where_is_zero[0][add_ix], S_where_is_zero[1][add_ix]] = w_add
    
    S_drop = np.zeros((p,p))
    S_drop[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]] = S[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]]
    
    S_new = S + S_add - S_drop
    S_new0 = S_new.copy()
    
    S_new = S_new + S_new.T
    
    vec_div = 1.5*np.sum(np.absolute(S_new), axis = 1)[:,None]    
    for i in range(p):
        if vec_div[i]: 
            # only when the absolute value of the vector is not zero do the standardization
            S_new[i,:] = S_new[i,:]/vec_div[i]
    # S_new = S_new/(1.5*np.sum(np.absolute(S_new), axis = 1)[:,None])
    A_new = (S_new + S_new.T)/2 + np.matrix(np.eye(p))
    
    # check PD
    np.all(alg.eigvals(A_new) > 0)   
    return S_new0, A_new

def getCov(A):
    A_inv = alg.inv(A) # inverse of A
    p = A.shape[0]
#    Sigma = 0.6*(np.triu(np.ones((p,p)),1) + np.tril(np.ones((p,p)),-1)) + np.matrix(np.eye(p))
    Sigma = np.zeros((p,p))
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
#            Sigma[i,j] = Sigma[i,j]*A_inv[i,j]/np.sqrt(A_inv[i,i]*A_inv[j,j])
            Sigma[i,j] = A_inv[i,j]/np.sqrt(A_inv[i,i]*A_inv[j,j])
    return Sigma

def genEmpCov(samples, useKnownMean = False, m = 0):
    # input samples p by n
    # size = p
    # samplesperstep = n
    size, samplesPerStep = samples.shape
    if useKnownMean == False:
        m = np.mean(samples, axis = 1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:,i]
        empCov = empCov + np.outer(sample - m, sample -m)
    empCov = empCov/samplesPerStep
    return empCov

def getF1(S0, S1):
    # S0 is the true graph and S1 is the estimated graph
    S_true, S_est = S0.copy(), S1.copy()
    np.fill_diagonal(S_true, 0)
    np.fill_diagonal(S_est, 0)
    
    # number of detected edges on off diagonal 
    D = np.where(S_est != 0)[0].shape[0]
    # number of true edges on off diagonal
    T = np.where(S_true != 0)[0].shape[0]
    
    # number of true edges detected
    TandD = float(np.where(np.logical_and(S_true, S_est))[0].shape[0])
    
#    print TandD
    if D: 
        P = TandD/D
    else:
        print('No edge detected on off diagonal, precision is zero')
        P = 0.
    R = TandD/T
    
    if P+R:
        F1 = 2*P*R/(P+R)
    else:
        F1 = 0.
    return P, R, F1

def getAIC(S_est, S_previous, empCov, ni):
#    S_diff = (S_est - S_previous)  
#    S_diff = S_diff - np.diag(np.diag(S_diff))
#    ind = (S_diff < 1e-2) & (S_diff > - 1e-2)
#    S_diff[ind] = 0    
#    K = np.count_nonzero(S_diff)
    ind = (S_est < 1e-2) & (S_est > - 1e-2)
    S_est[ind] = 0
    ind = (S_previous < 1e-2) & (S_previous > - 1e-2)
    S_previous[ind] = 0
    
    K = float(np.where(np.logical_and((S_est!=0) != (S_previous!=0), S_est!=0) == True)[0].shape[0])
    # K = float(np.where(np.logical_and((S_est>0) != (S_previous>0), S_est>0) == True)[0].shape[0])
    #loglik = ni*(np.log(alg.det(S_est)) - np.trace(np.dot(S_est, empCov)))
    loglik = np.log(alg.det(S_est)) - np.trace(np.dot(S_est, empCov))
    #print(-loglik)
    #print(K)
    AIC = -loglik + K
    return AIC

def indicesOfExtremeValue(arr, set_length, choice):
    if (choice == 'max'):
        index = np.argmax(arr)
    elif (choice == 'min'):
        index = np.argmin(arr)
    else:
        print 'invalid argument, choose max or min'
    index_x = index/set_length
    index_y = index - (index_x)*set_length
    return index, index_x, index_y

def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.
    Parameters
    ----------
    emp_cov : 2D array, (n_features, n_features)
        The sample covariance matrix
    Notes
    -----
    This results from the bound for the all the Lasso that are solved
    in GraphLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.
    """
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))
#------------------------------------------- End defining private functions ----------------------------------------------------
    
#---------------------------------------- Generating basic structures ----------------------------------------------

p0 = 20
p1 = 20
ni = 50
p_vec = [p0, p1, p1, p1, p1] # node size for common structures and for 4 classes
d_vec = [1, 1, 1, 1, 1] # out degree

n_block = len(p_vec)

# A_list: n_block graphs stored in Ab_list
Ab_list = []
S_list = []
for i in range(n_block):
    gG = getGraph(p_vec[i], d_vec[i])
    Ab_list.append(gG[1])
    S_list.append(gG[0])

# common graph structure 
A0 = Ab_list[0]
S0 = S_list[0]

n_change = 2
A_T = getGraphatT_Shuheng(S0, A0, n_change)[1]

len_t = 11
A0_list = [lam*A_T+(1-lam)*A0 for lam in np.linspace(0, 1, len_t)]

A1_list = Ab_list[1:] # A1_list: different graph structures for different classes

# covariance matrices
# using my own function
C0_list = [getCov(item) for item in A0_list] # common part
C1_list = [getCov(item) for item in A1_list] # different part for class

# check if all of those covariance matrices are PD
for C in C0_list:
    print(np.all(alg.eigvals(C) > 0))

for C in C1_list:
    print(np.all(alg.eigvals(C) > 0))
# directly inverse
#C0_list = [alg.inv(item) for item in A0_list] # common part
#C1_list = [alg.inv(item) for item in A1_list] # different part for class

len_class = len(C1_list)
p = sum(p_vec)
#---------------------------------------- End generating basic structures -------------------------------------------------

#------------------------------------- Generating graphs, covariances, multivariate normal observarions -----------------------------------------
A_list = []
C_list = [] # first dim is time second dim is group
X_list = []

#ml_glassocv = cov.GraphLassoCV(assume_centered=True)
#Theta_glassocv_list = []
for class_ix in range(len_class):
    A_c = []
    C_c = []
    X_c = []
    #Theta_t = []
    for time_ix in range(len_t):
        #print class_ix
        #print block_diag(*A1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*5))
        A = block_diag(A0_list[time_ix], block_diag(*A1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*p1)))
        C = block_diag(C0_list[time_ix], block_diag(*C1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*p1)))
        X = np.random.multivariate_normal(mean = np.zeros(p), cov = C, size = ni)
        #ml_glassocv.fit(X)
        #Theta = ml_glassocv.get_precision()
        A_c.append(A)
        C_c.append(C)
        X_c.append(X)
        #Theta_t.append(Theta)
    A_list.append(A_c)
    C_list.append(C_c)
    X_list.append(X_c)
    #Theta_glassocv_list.append(Theta_t)
    
f = open("mydata.pkl", 'wb')
pickle.dump(X_list, f)
pickle.dump(A_list, f)
f.close()
#-------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------- graphical lasso ----------------------------------------------
set_length = 51
#F1_result_list = []
#for class_ix in range(len_class):
#    F1_result_c = []
#    for time_ix in range(len_t):
#        F1_result_t = []
#        cov_last = None
#        alpha_max_ct = alpha_max(genEmpCov(X_list[class_ix][time_ix].T))
#        alpha_set = np.logspace(np.log10(alpha_max_ct*5e-2), np.log10(alpha_max_ct), set_length)
#        for alpha in alpha_set:
#            emp_cov = genEmpCov(X_list[class_ix][time_ix].T)
#            ml_glasso = cov.graph_lasso(emp_cov, alpha, cov_init=cov_last, max_iter = 500) 
#            cov_last = ml_glasso[0]
#            A_est = ml_glasso[1]
#            F1_result_t.append(getF1(A_list[class_ix][time_ix], A_est))
#        F1_result_c.append(F1_result_t)
#    F1_result_list.append(F1_result_c)
    
Theta_glasso_list = [] # 1 dim: class_ix; 2 dim: time_ix; 3 dim: alpha
for class_ix in range(len_class):
    Theta_glasso_c = []
    for time_ix in range(len_t):
        Theta_glasso_t = []
        cov_last = None
        alpha_max_ct = alpha_max(genEmpCov(X_list[class_ix][time_ix].T))
        alpha_set = np.logspace(np.log10(alpha_max_ct*5e-2), np.log10(alpha_max_ct), set_length)
        for alpha in alpha_set:
            emp_cov = genEmpCov(X_list[class_ix][time_ix].T)
            ml_glasso = cov.graph_lasso(emp_cov, alpha, cov_init=cov_last, max_iter = 500) 
            cov_last = ml_glasso[0]
            Theta_glasso_t.append(ml_glasso[1])
        Theta_glasso_c.append(Theta_glasso_t)
    Theta_glasso_list.append(Theta_glasso_c)
#-----------------------------------------------------------------------------------------------------
    
#----------------------------------- Shuheng's method ------------------------------------------------
beta = 0
indexOfPenalty = 1
set_length = 51
alpha_set = np.logspace(np.log10(0.9*5e-2), np.log10(0.9), set_length)
Theta_Shuheng_list = [] # first dim: class_ix; second dim: alpha; third dim: time_ix

for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    ThetaSet_c = []
    for alpha in alpha_set:
        ThetaSet = mtvgl.TVGL(X_concat, ni, alpha, beta, indexOfPenalty, useKernel = True, verbose=False)
        ThetaSet_c.append(ThetaSet)
    Theta_Shuheng_list.append(ThetaSet_c)

# F1 score
#F1_result_Shuheng_list = []
#for class_ix in range(len_class):
#    F1_result_c = []
#    for time_ix in range(len_t):
#        F1_result_t = []
#        for alpha_ix in range(set_length):
#            F1_result_t.append(getF1(A_list[class_ix][time_ix], Theta_Shuheng_list[class_ix][alpha_ix][time_ix]))
#        F1_result_c.append(F1_result_t)
#    F1_result_Shuheng_list.append(F1_result_c)
#-----------------------------------------------------------------------------------------------------
    
#----------------------------------- My own method ---------------------------------------------------
indexOfPenalty = 1
set_length = 51
alpha_set = np.logspace(np.log10(0.9*5e-2), np.log10(0.9), set_length)
beta_set = np.logspace(-2, 0.5, set_length)

X_concat_list = []
for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    X_concat_list.append(X_concat)

Theta_mymethod_list = [] # 1 dim: beta; 2 dim: alpha; 3 dim: time_ix; 4 dim: class_ix
for beta in beta_set:
    Theta_mymethod_beta = []
    for alpha in alpha_set:
        result = mtvgl.myTVGL(X_concat_list, ni, alpha, beta, indexOfPenalty, useKernel = True, verbose=False)    
        Theta_mymethod_beta.append(result)
    Theta_mymethod_list.append(Theta_mymethod_beta)
#-----------------------------------------------------------------------------------------------------
    
#------------------------------------ Paper's method -------------------------------------------------
indexOfPenalty = 1
set_length = 51
alpha_set = np.logspace(np.log10(0.9*5e-2), np.log10(0.9), set_length)
beta_set = np.logspace(-2, 0.5, set_length)

Theta_paper_list = [] # 1 dim: class_ix; 2 dim: alpha; 3 dim: beta; 4 dim: time_ix
for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    ThetaSet_c = []
    for alpha in alpha_set:
        ThetaSet_c_alpha = []
        for beta in beta_set:
            ThetaSet = tvgl.TVGL(X_concat, ni, alpha, beta, indexOfPenalty, verbose=False)
            ThetaSet_c_alpha.append(ThetaSet)
        ThetaSet_c.append(ThetaSet_c_alpha)
    Theta_paper_list.append(ThetaSet_c)
#-----------------------------------------------------------------------------------------------------

    