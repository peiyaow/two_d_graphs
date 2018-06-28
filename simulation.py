#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('..')+'/TVGL')

import numpy as np
from random import *
import snap
import numpy.linalg as alg
from scipy.linalg import block_diag
import sklearn.covariance as cov
import TVGL as tvgl
import myTVGL as mtvgl

# generate time-varying graph using sliding window

#---------------------------------------------- Define private functions -----------------------------------------------------

def getGraph(p,d):
    # construct a graph from scale free distribution
    # paper: The joint graphical lasso for inverse covarianceestimation across multiple classes
    # reference: https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/rssb.12033
    # p: number of nodes
    # d: out degree of each node
    Rnd = snap.TRnd()
    UGraph = snap.GenPrefAttach(p, d, Rnd)
    S = np.zeros((p,p))
    for EI in UGraph.Edges():
    # generate a random number in (-0.4, -0.1)U(0.1,0.4)
    # method: https://stats.stackexchange.com/questions/270856/how-to-randomly-generate-random-numbers-in-one-of-two-intervals
        r = np.random.uniform(0, 0.6)
        S[EI.GetSrcNId(), EI.GetDstNId()] = r-0.4 if r < 0.3 else r-0.2 # assign values to edges
    
    S0 = S.copy()
    # compute the covariance
    S =  S + S.T 
    S = S/(1.5*np.sum(np.absolute(S), axis = 1)[:,None])
    A = (S + S.T)/2 + np.matrix(np.eye(p))
    # check if A is PD
    # np.all(alg.eigvals(A) > 0) 
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
    
    drop_ix = sample(range(n_offdiag_nonzero), n_change)
    add_ix = sample(range(n_offdiag_zero), n_change) 
    
    w_add = []
    for i in range(n_change):
        r = np.random.uniform(0, 0.6)
        w_add.append(r-0.4 if r < 0.3 else r-0.2)
    
    S_add = np.zeros((p,p))
    S_add[S_where_is_zero[0][add_ix], S_where_is_zero[1][add_ix]] = w_add
    
    S_drop = np.zeros((p,p))
    S_drop[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]] = S[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]]
    
    S_new = S + S_add - S_drop
    S_new = S_new + S_new.T
    S_new = S_new/(1.5*np.sum(np.absolute(S_new), axis = 1)[:,None])
    A_new = (S_new + S_new.T)/2 + np.matrix(np.eye(p))
    
    # check PD
    np.all(alg.eigvals(A_new) > 0) 
    
    return S_new, A_new

def getCov(A):
    A_inv = alg.inv(A) # inverse of A
    p = A.shape[0]
    Sigma = 0.6*(np.triu(np.ones((p,p)),1) + np.tril(np.ones((p,p)),-1)) + np.matrix(np.eye(p))
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
            Sigma[i,j] = Sigma[i,j]*A_inv[i,j]/np.sqrt(A_inv[i,i]*A_inv[j,j])
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
    # for one S_true and one S_est p by p    
    S_true, S_est = S0.copy(), S1.copy()
    np.fill_diagonal(S_true, 0)
    np.fill_diagonal(S_est, 0)
    
    D = np.where(S_est != 0)[0].shape[0]
    T = np.where(S_true != 0)[0].shape[0]
    
    print D
    print T
    
    TandD = float(np.where(np.logical_and(S_true, S_est))[0].shape[0])
    
    print TandD
#    P = TandD/D
#    R = TandD/T
#    return 2*P*R/(P+R)
    return

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
#------------------------------------------- End defining private functions ----------------------------------------------------


#---------------------------------------- Generating basic structures ----------------------------------------------
p_vec = [20, 5, 5, 5, 5] # node size for common structures and for 4 classes
d_vec = [4, 2, 2, 2, 2] # out degree

# A_list: 5 graphs stored in 
Ab_list = []
for i in range(5):
    Ab_list.append(getGraph(p_vec[i], d_vec[i])[1])

A0 = Ab_list[0] # common graph structure 
A1_list = Ab_list[1:] # A1_list: different graph structures for different classes

# A0_list: common graph structures at different time points
# using the idea of sliding window
step = 5 
# step size for time changing 
A0_list = []
for i in range(Ab_list[0].shape[0]-step+1):
    Id = np.matrix(np.eye(p_vec[0]))
    Id[i:(i+step), i:(i+step)] = Ab_list[0][i:(i+step), i:(i+step)]
    A0_list.append(Id)

# covariance matrices
C0_list = [getCov(item) for item in A0_list] # common part
C1_list = [getCov(item) for item in A1_list] # different part for class

len_t = len(C0_list)
len_class = len(C1_list)
ni = 500
p = sum(p_vec)
#---------------------------------------- End generating basic structures -------------------------------------------------


##---------------------------------------- Generating Covariance -----------------------------------------------------------
#C_list = [] # first dim is time second dim is group
#for time_ix in range(len(C0_list)):
#    C_t = []
#    for class_ix in range(len(C1_list)):
#        C_t.append(block_diag(C0_list[time_ix], block_diag(*C1_list[:(len(C1_list) - class_ix)]), np.matrix(np.eye(class_ix*5))))
#    C_list.append(C_t)
##--------------------------------------------------------------------------------------------------------------------------
#
##------------------------------------------- Generating multivariate normal X -----------------------------------------
#X_list = []
#
#for time_ix in range(len(C_list)):
#    X_t = []
#    for class_ix in range(len(C_list[0])):
#        X = np.random.multivariate_normal(mean = np.zeros(p), cov = C_list[time_ix][class_ix], size = ni)
#        # print(X.shape)
#        # X.shape: ni by p
#        X_t.append(X)
#    X_list.append(X_t)
##----------------------------------------------------------------------------------------------------------------------
#
##-------------------------------------- Estimating graph using graphical lasso ----------------------------------------
## sckit learn to get inverse covariance matrix
#ml_glasso = cov.GraphLasso(alpha = 0.1)
#Theta_glasso_list = []
#for time_ix in range(len(C_list)):
#    Theta_t = []
#    for class_ix in range(len(C_list[0])):
#        X = X_list[time_ix][class_ix]
#        ml_glasso.fit(X)
#        Theta = ml_glasso.get_precision()
#        Theta_t.append(Theta)
#    Theta_glasso_list.append(Theta_t)
##----------------------------------------------------------------------------------------------------------------------
#
##---------------------------------- Estimating using CV graphical lasso -----------------------------------------------
#ml_glassocv = cov.GraphLassoCV(assume_centered=True)
#Theta_glassocv_list = []
#for time_ix in range(len(C_list)):
#    Theta_t = []
#    for class_ix in range(len(C_list[0])):
#        X = X_list[time_ix][class_ix]
#        ml_glassocv.fit(X)
#        Theta = ml_glassocv.get_precision()
#        # print(ml_glassocv.cv_alphas_)
#        # print(ml_glassocv.alphas_)
#        Theta_t.append(Theta)
#    Theta_glassocv_list.append(Theta_t)
##----------------------------------------------------------------------------------------------------------------------
    
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
        print class_ix
        print block_diag(*A1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*5))
        A = block_diag(A0_list[time_ix], block_diag(*A1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*5)))
        C = block_diag(C0_list[time_ix], block_diag(*C1_list[:(len_class - class_ix)]), np.matrix(np.eye(class_ix*5)))
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
#-------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------- Graphical Lasso ---------------------------------
ml_glassocv = cov.GraphLassoCV(assume_centered=False)
Theta_glassocv_list = []
for class_ix in range(len_class):
    Theta_c = []
    for time_ix in range(len_t):
        ml_glassocv.fit(X_list[class_ix][time_ix])
        Theta = ml_glassocv.get_precision()
        Theta_c.append(Theta)
    Theta_glassocv_list.append(Theta_c)
    
# F1 score for graphical lasso
for class_ix in range(len_class):
    for time_ix in range(len_t):
        getF1(A_list[class_ix][time_ix], Theta_glassocv_list[class_ix][time_ix])
#        print(getF1(A_list[class_ix][time_ix], Theta_glassocv_list[class_ix][time_ix]))

# alpha_max in GraphLassoCV gives the largest penalty alpha for cross validation
#----------------------------------------------------------------------------------        



#------------------------- use the original method TVGL from the paper -------------------------------------
# fix tuning parameter
lamb = 2
beta = 12
indexOfPenalty = 3
Theta_paper_list = []
for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    ThetaSet = tvgl.TVGL(X_concat, ni, lamb, beta, indexOfPenalty, verbose=True)
    Theta_paper_list.append(ThetaSet)

# F1 score
for class_ix in range(len_class):
    print class_ix
    for time_ix in range(len_t):
        print(getF1(A_list[class_ix][time_ix], Theta_paper_list[class_ix][time_ix]))       
#-----------------------------------------------------------------------------------------------------------
# indexOfPenalty = 4

#------------------------------------- selection criteria AIC -------------------------------------------
set_length = 10
alpha_set = np.linspace(0.2, 1 , set_length)
beta_set = np.logspace(-2, 0.5, set_length)
ni = 50

alpha_selected_list = []
beta_selected_list = []
for indexOfPenalty in np.arange(1, 6):
    alpha_ixPen_list = []
    beta_ixPen_list = []
    for class_ix in range(len_class):
        AIC_list = []
        for alpha in alpha_set:
            for beta in beta_set:
                EmpCov_list = [genEmpCov(myX.T) for myX in X_list[class_ix]]
                X_concat = np.concatenate(X_list[class_ix])
                ThetaSet = tvgl.TVGL(X_concat, ni, alpha, beta, indexOfPenalty, verbose=False)
                ThetaSet.insert(0, np.zeros((p,p)))
                AIC_ab_list = [getAIC(ThetaSet[ix+1], ThetaSet[ix], EmpCov_list[ix], ni) for ix in range(len(ThetaSet)-1)]
                AIC_list.append(sum(AIC_ab_list))
    index, index1, index2 = indicesOfExtremeValue(AIC_list, set_length, 'min')
    alpha_ixPen_list.append(alpha_set[index1])
    beta_ixPen_list.append(beta_set[index2])            
alpha_selected_list.append(alpha_ixPen_list)
beta_selected_list.append(beta_ixPen_list)
#---------------------------------------------------------------------------------------------------------

#----------------------------------- Kernel Method Estimation --------------------------------------------
lamb = .2
beta = .5
indexOfPenalty = 3
Theta_paper_list = []
for class_ix in range(len_class):
    X_concat = np.concatenate(X_list[class_ix])
    ThetaSet = mtvgl.TVGL(X_concat, ni, lamb, beta, indexOfPenalty, useKernel = True, verbose=True)
    Theta_paper_list.append(ThetaSet)

# F1 score
for class_ix in range(len_class):
    print class_ix
    for time_ix in range(len_t):
        print(getF1(A_list[class_ix][time_ix], Theta_paper_list[class_ix][time_ix])) 
#---------------------------------------------------------------------------------------------------------
        
set_length = 10
alpha_set = np.logspace(-1, 1 , set_length)    

Theta_glasso_list = []
for class_ix in range(len_class):
    Theta_c = []
    for time_ix in range(len_t):
        for alpha in alpha_set:
            ml_glasso = cov.GraphLasso(alpha, assume_centered=False)
            ml_glasso.fit(X_list[class_ix][time_ix])
            Theta = ml_glasso.get_precision()
            Theta_c.append(Theta)
        Theta_glasso_list.append(Theta_c)


set_length = 51
alpha_set = np.logspace(-1, .5 , set_length)  
Theta_c = []
class_ix = 0
time_ix = 0
for alpha in alpha_set:
    ml_glasso = cov.GraphLasso(alpha, assume_centered=False)
    ml_glasso.fit(X_list[class_ix][time_ix])
    Theta = ml_glasso.get_precision()
    getF1(A_list[0][0], Theta)
    Theta_c.append(Theta)
    
# true positives vs false postives


   



    

    
    
    
    