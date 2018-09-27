#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:57:47 2018

@author: MonicaW
"""
import sys
#import os
# sys.path.append(os.path.abspath('..')+'/TVGL')
sys.path.append('/nas/longleaf/home/peiyao/proj2/TVGL')

import numpy as np
import random
import os
# import snap
import numpy.linalg as alg
from scipy.linalg import block_diag
import sklearn.covariance as cov

import networkx as nx
from scipy import sparse

import multiprocessing as mp
from posdef import *
import itertools

# from scipy.linalg import block_diag
# import TVGL as tvgl
# import myTVGL as mtvgl

#---------------------------------------------- Define private functions -----------------------------------------------------
#def getGraph_snap(p,d):
#    # construct a graph from scale free distribution
#    # paper: The joint graphical lasso for inverse covariance estimation across multiple classes
#    # reference: https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/rssb.12033
#    # another paper: Partial Correlation Estimation by Joint Sparse Regression Models
#    # p: number of nodes
#    # d: out degree of each node
#    Rnd = snap.TRnd()
#    UGraph = snap.GenPrefAttach(p, d, Rnd)
#    S = np.zeros((p,p))
#    for EI in UGraph.Edges():
#    # generate a random number in (-1, -0.5)U(0.5,1)
#    # method: https://stats.stackexchange.com/questions/270856/how-to-randomly-generate-random-numbers-in-one-of-two-intervals
#        r = np.random.uniform(0, 1.)
##        print r
##        print EI.GetSrcNId(), EI.GetDstNId()
#        S[EI.GetSrcNId(), EI.GetDstNId()] = r-1. if r < 0.5 else r  # assign values to edges
#    
#    S0 = S.copy() 
#    # orginal half graph without standardizing it into a PD matrix
##    S =  S + S.T
#    # the whole row could be all zeros
#    vec_div = 1.5*np.sum(np.absolute(S), axis = 1)[:,None]    
##    print(vec_div)
#    for i in range(p):
##        print(vec_div[i])
#        if vec_div[i]: 
#            # only when the absolute value of the vector is not zero do the standardization
#            S[i,:] = S[i,:]/vec_div[i]
##        else:
##            print 'all zero vectors exists!'
##    S = S/(1.5*np.sum(np.absolute(S), axis = 1)[:,None])
#    A = (S + S.T)/2 + np.matrix(np.eye(p))
#    # check if A is PD
#    print(np.all(alg.eigvals(A) > 0))
#    #print(alg.eigvals(A))
#    return S0, A

def getGraph(p,d):
#    # construct a graph from scale free distribution
#    # paper: The joint graphical lasso for inverse covariance estimation across multiple classes
#    # reference: https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/rssb.12033
#    # another paper: Partial Correlation Estimation by Joint Sparse Regression Models
#    # p: number of nodes
#    # d: out degree of each node
    S = nx.barabasi_albert_graph(p, d)  
    S = nx.adjacency_matrix(S)
    S = sparse.triu(S)
    row_ix, col_ix = sparse.find(S)[0:2]
    n_nonzero = len(sparse.find(S)[2])
    S = S.todense().astype(float)
    S = np.array(S)
    S0 = S.copy()
    for i in range(n_nonzero):
        r = np.random.uniform(0, 1.)
        S[row_ix[i], col_ix[i]] = r-1. if r < 0.5 else r
        # r = np.random.uniform(-.2, 0.4)
        # S[row_ix[i], col_ix[i]] = r - .2 if r < 0.1 else r

    #add
    S = S + np.transpose(S)
    #add end

    vec_div = 1.5*np.sum(np.absolute(S), axis = 1)[:,None]
    for i in range(p):
        if vec_div[i]: 
            # only when the absolute value of the vector is not zero do the standardization
            S[i,:] = S[i,:]/vec_div[i]
    A = (S + S.T)/2 + np.matrix(np.eye(p))
    # check if A is PD
    # print(np.all(alg.eigvals(A) > 0))
    #print(alg.eigvals(A))
    return S0, A
    
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
        r = np.random.uniform(0, 1.)
        w_add.append(r-1. if r < 0.5 else r)
    
    S_add = np.zeros((p,p))
    S_add[S_where_is_zero[0][add_ix], S_where_is_zero[1][add_ix]] = w_add
    
    S_drop = np.zeros((p,p))
    S_drop[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]] = S[S_where_is_not_zero[0][drop_ix], S_where_is_not_zero[1][drop_ix]]
    
    S_new = S + S_add - S_drop
    S_new0 = S_new.copy()

    #add
    S_new = S_new + S_new.T
    #add end
    
    vec_div = 1.5*np.sum(np.absolute(S_new), axis = 1)[:,None]    
    for i in range(p):
        if vec_div[i]: 
            # only when the absolute value of the vector is not zero do the standardization
            S_new[i,:] = S_new[i,:]/vec_div[i]
    # S_new = S_new/(1.5*np.sum(np.absolute(S_new), axis = 1)[:,None])
    A_new = (S_new + S_new.T)/2 + np.matrix(np.eye(p))
    
    # check PD
    # print(np.all(alg.eigvals(A_new) > 0))
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

def getF1(S0, S1, include_diagonal = True):
    # S0 is the true graph and S1 is the estimated graph
    S_true, S_est = S0.copy(), S1.copy()
    if not include_diagonal:
        np.fill_diagonal(S_true, 0)
        np.fill_diagonal(S_est, 0)
    
    # number of detected edges
    D = np.where(S_est != 0)[0].shape[0]
    # number of true edges
    T = np.where(S_true != 0)[0].shape[0]
    
    # number of true edges detected
    TandD = float(np.where(np.logical_and(S_true, S_est))[0].shape[0])
    
#    print TandD
    if D: 
        P = TandD/D
    else:
#         print('No edge detected on off diagonal, precision is zero')
        P = np.nan
    R = TandD/T
    
    if P+R:
        F1 = 2*P*R/(P+R)
    else:
        F1 = np.nan
    return P, R, F1


def getF1_diagonal(S0, S1):
    # S0 is the true graph and S1 is the estimated graph
    S_true, S_est = S0.copy(), S1.copy()
    #np.fill_diagonal(S_true, 0)
    #np.fill_diagonal(S_est, 0)

    # number of detected edges on off diagonal
    D = np.where(S_est != 0)[0].shape[0]
    # number of true edges on off diagonal
    T = np.where(S_true != 0)[0].shape[0]

    # number of true edges detected
    TandD = float(np.where(np.logical_and(S_true, S_est))[0].shape[0])

    #    print TandD
    if D:
        P = TandD / D
    else:
        #         print('No edge detected on off diagonal, precision is zero')
        P = np.nan
    R = TandD / T

    if P + R:
        F1 = 2 * P * R / (P + R)
    else:
        F1 = np.nan
    return P, R, F1

def getPD(Theta_array, A_list, class_ix = 0, time_ix = 5):
    set_length = Theta_array.shape[2]
    precision_list = []
    recall_list = []
    for i in range(set_length):
        P,R,F1 = getF1(A_list[class_ix][time_ix], Theta_array[class_ix][time_ix][i])
        precision_list.append(P)
        recall_list.append(R)
    return (precision_list, recall_list)

# --- plotting tools ---
def PD_array(Theta_array, A_list, class_ix):
    # input: Theta_array: beta by class by time by alpha by p by p
    # output: result2: array beta by 2(P or R) by alpha
    len_beta = Theta_array.shape[0]
    P_list = []
    R_list = []
    for i in range(len_beta):
       P, R = getPD(Theta_array[i], A_list, class_ix)
       P_list.append(P)
       R_list.append(R)
    result2 = np.array([P_list, R_list]) # 2(P or R) by beta by alpha
    result2 = np.transpose(result2, [1,0,2]) # beta by 2(P or R) by alpha
    return result2

def PD_array_simple(Theta_array_simple, A_list, class_ix, time_ix):
    # Theta_array: class by time by alpha by p by p
    P_list, R_list = getPD(Theta_array_simple, A_list, class_ix, time_ix)
    result = np.array([P_list, R_list]) # 2(P or R) by alpha
    return result

# set working directory as TVGL_result
def get_average_array(name):
    glasso_P_array = np.array([]).reshape(0,51)
    glasso_R_array = np.array([]).reshape(0,51)
    for i in range(50):
#        path = "./glasso/glasso"+str(i)+".npy"    
        path = "./"+name+"/"+name+str(i)+".npy" 
        result = np.load(path)
        glasso_P_array = np.concatenate((glasso_P_array, result[0, None]), axis = 0)
        glasso_R_array = np.concatenate((glasso_R_array, result[1, None]), axis = 0)

    P_inx = np.logical_and(glasso_P_array!=0,  np.logical_not(np.isnan(glasso_P_array)))
    R_inx = np.logical_and(glasso_R_array!=0,  np.logical_not(np.isnan(glasso_R_array)))    
    glasso_P_array[np.logical_not(P_inx)] = 0
    glasso_R_array[np.logical_not(R_inx)] = 0    
    glasso_P_array_average = np.sum(glasso_P_array*P_inx, axis=0)/np.sum(P_inx, axis=0)
    glasso_R_array_average = np.sum(glasso_R_array*R_inx, axis=0)/np.sum(R_inx, axis=0)
    return np.array((glasso_P_array_average, glasso_R_array_average))

def get_average_array_ix(name, ix):
    glasso_P_array = np.array([]).reshape(0,51)
    glasso_R_array = np.array([]).reshape(0,51)
    for i in range(50):
        #path = "./glasso/glasso"+str(i)+".npy"    
        path = "./"+name+"/"+name+str(i)+".npy" 
        if os.path.exists(path):
            result = np.load(path)
            glasso_P_array = np.concatenate((glasso_P_array, result[ix][0, None]), axis = 0)
            glasso_R_array = np.concatenate((glasso_R_array, result[ix][1, None]), axis = 0)

    P_inx = np.logical_and(glasso_P_array!=0,  np.logical_not(np.isnan(glasso_P_array)))
    R_inx = np.logical_and(glasso_R_array!=0,  np.logical_not(np.isnan(glasso_R_array)))    
    glasso_P_array[np.logical_not(P_inx)] = 0
    glasso_R_array[np.logical_not(R_inx)] = 0    
    glasso_P_array_average = np.sum(glasso_P_array*P_inx, axis=0)/np.sum(P_inx, axis=0)
    glasso_R_array_average = np.sum(glasso_R_array*R_inx, axis=0)/np.sum(R_inx, axis=0)
    return np.array((glasso_P_array_average, glasso_R_array_average))
# -----------------------
    
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
        print('invalid argument, choose max or min')
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

def simulate_data(p0 = 20, p1 = 20, d0 = 1, d1 = 1, ni = 50, n_change = 2, len_t = 11):
    #---------------------------------------- Generating basic structures ----------------------------------------------
    p_vec = [p0, p1, p1, p1, p1] # node size for common structures and for 4 classes
    p = sum(p_vec)
    
    d_vec = [d0, d1, d1, d1, d1] # out degree
    
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
    
    A_T = getGraphatT_Shuheng(S0, A0, n_change)[1]
    
    A0_list = [lam*A_T+(1-lam)*A0 for lam in np.linspace(0, 1, len_t)]
    
    A1_list = Ab_list[1:] # A1_list: different graph structures for different classes
    
    # covariance matrices
    # using my own function
    C0_list = [getCov(item) for item in A0_list] # common part
    C1_list = [getCov(item) for item in A1_list] # different part for class
    len_class = len(C1_list)
    
    # check if all of those covariance matrices are PD
    #for C in C0_list:
    #    print(np.all(alg.eigvals(C) > 0))
    #
    #for C in C1_list:
    #    print(np.all(alg.eigvals(C) > 0))
    
    # direct inverse
    #C0_list = [alg.inv(item) for item in A0_list] # common part
    #C1_list = [alg.inv(item) for item in A1_list] # different part for class
    
    
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
    return A_list, C_list, X_list

def simulate_data_xie(p = 100, d = 3, n_vec = [50, 50, 50, 50], ni = 50, n_change = 3, len_t = 50, K = 4):
    # S_list = [] # upper triangle
    A_list = [] # Omega graph 
    C_list = [] # Covariance
    for k in range(K+1):
        gG = getGraph(p, d)
        S = gG[0]
        A = gG[1]
        A_T = getGraphatT_Shuheng(S, A, n_change)[1]
        A_T_list = [lam*A_T+(1-lam)*A for lam in np.linspace(0, 1, len_t)] # Omega
        C_T_list = [getCov(item) for item in A_T_list] # Cov: 0+class time
        
        A_list.append(A_T_list)
        C_list.append(C_T_list)
        
    A_array = np.array(A_list) # class time p p 
    A_add_list = [] # class time p p
    for k in range(1,K+1):
        A_k_list = []
        for time_ix in range(len_t):
            O_0_inv = alg.inv(A_array[0][time_ix])
            O_k_inv = alg.inv(A_array[k][time_ix])
            A_k_list.append(alg.inv(O_0_inv+O_k_inv))
        A_add_list.append(A_k_list)
        
    Y_list = []
    for time_ix in range(len_t):
        C0 = C_list[0][time_ix] 
        Y_t = []
        Z = np.random.multivariate_normal(mean = np.zeros(p), cov = C0, size = ni)
        for k in range(1,K+1):
            Ck = C_list[k][time_ix]
            X = np.random.multivariate_normal(mean = np.zeros(p), cov = Ck, size = n_vec[k-1])
            Y = X+Z
            Y_t.append(Y)
        Y_list.append(Y_t)
        
    Y_array = np.array(Y_list) # time class n p
#    Y_array = np.transpose(Y_array, [0, 2, 1, 3]) # time n class p
#    Y_array = np.reshape(Y_array, [len_t, n_vec[0], K*p]) # time n Kp
        
    return A_list, A_add_list, C_list, Y_array

def simulate_data_2dgraph(p = 100, d = 3, n_vec = [50, 50, 50, 50], ni = 50, n_change = 3, len_t = 51, K = 4):
    # S_list = [] # upper triangle
    A_list = [] # Omega graph 
    C_list = [] # Covariance
    for k in range(K):
        gG = getGraph(p, d)
        S = gG[0]
        A = gG[1]
        A_T = getGraphatT_Shuheng(S, A, n_change)[1]
        A_T_list = [lam*A_T+(1-lam)*A for lam in np.linspace(0, 1, len_t)] # Omega
        C_T_list = [getCov(item) for item in A_T_list] # Cov: 0+class time
        A_list.append(A_T_list)
        C_list.append(C_T_list)
        
    A_array = np.array(A_list) # class time p p 
    A_add_list = [] # class time p p
    
    gG = getGraph(p, d)
    A0 = gG[1]
    C0 = getCov(A0)
    
    for k in range(K):
        A_k_list = []
        for time_ix in range(len_t):
            O_0_inv = alg.inv(A0)
            O_k_inv = alg.inv(A_array[k][time_ix])
            A_k_list.append(alg.inv(O_0_inv+O_k_inv))
        A_add_list.append(A_k_list)
        
    Y_list = []
    for k in range(K):
#        C0 = C_list[0][time_ix] 
        Y_k = []
        Z = np.random.multivariate_normal(mean = np.zeros(p), cov = C0, size = n_vec[k])
        for time_ix in range(len_t):
            Ck = C_list[k][time_ix]
            X = np.random.multivariate_normal(mean = np.zeros(p), cov = Ck, size = n_vec[k])
            Y = X+Z
            Y_k.append(Y)
        Y_list.append(Y_k)
#    Y_array = np.array(Y_list) # class time n p
#    Y_array = np.transpose(Y_array, [0, 2, 1, 3]) # time n class p
#    Y_array = np.reshape(Y_array, [len_t, n_vec[0], K*p]) # time n Kp
    return A_list, A_add_list, C_list, Y_list

def myglasso(X_list, ix_product, set_length):
    class_ix, time_ix = ix_product
    cov_last = None
    alpha_max_ct = alpha_max(genEmpCov(X_list[class_ix][time_ix].T))
    alpha_set = np.logspace(np.log10(alpha_max_ct*5e-2), np.log10(alpha_max_ct), set_length)
    result = []
    for alpha in alpha_set:
        emp_cov = genEmpCov(X_list[class_ix][time_ix].T)
        ml_glasso = cov.graph_lasso(emp_cov, alpha, cov_init=cov_last, verbose = True) 
        cov_last = ml_glasso[0]
        result.append(ml_glasso[1])
    return result

def Shuheng_method(X_list, ix_product, set_length, sigma, width = 5, knownMean = False, m = 0):
    class_ix, time_ix = ix_product
    cov_last = None
    alpha_max_ct = alpha_max(genEmpCov(X_list[class_ix][time_ix].T))
    alpha_set = np.logspace(np.log10(alpha_max_ct*5e-2), np.log10(alpha_max_ct), set_length)
    result = []
    for alpha in alpha_set:
        emp_cov = genEmpCov_kernel(time_ix, sigma, width, X_list[class_ix], knownMean, m)
        ml_glasso = cov.graph_lasso(emp_cov, alpha, cov_init=cov_last, verbose = True) 
        cov_last = ml_glasso[0]
        result.append(ml_glasso[1])
        print(alpha)
    return result

# each element in sample_set is n by p len is total length of timestamp
# total width = 2*width + 1
def genEmpCov_kernel(t_query, sigma, width, sample_set, knownMean = False, m=0):
    timesteps = sample_set.__len__()
#    print timesteps
    K_sum = 0
    S = 0
#    print(range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))))
    if knownMean != True:
        for j in range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))):         
            K =  np.exp(-np.square(t_query - j)/sigma)
            samplesPerStep = sample_set[j].shape[0]
            mean = np.mean(sample_set[j], axis = 0) # p by 1
#            print mean
            mean_tile = np.tile(mean, (samplesPerStep,1))
#            print mean_tile.shape
            S = S + K*np.dot((sample_set[j]- mean_tile).T, sample_set[j] - mean_tile)/samplesPerStep
            K_sum = K_sum + K
    else:
        for j in range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))):         
            K =  np.exp(-np.square(t_query - j)/sigma)
            samplesPerStep = sample_set[j].shape[0]
            S = S + K*np.dot((sample_set[j]-m).T, sample_set[j]-m)/samplesPerStep
            K_sum = K_sum + K
    S = S/K_sum    
    return S

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
 
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(mp.pool.Pool):
    Process = NoDaemonProcess
    
def EM_xie_time_varying(S_Y, p, K, set_length):
    # S_Y empirical covariance
    # set_length: length for lambdas
    S_0 = np.zeros([p, p]) # Sigma_0
    for m in range(K):
        for l in [i for i in range(K) if i != m]:
            S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
            S_0 = S_0 + S_ml
    S_0 = S_0/((K-1)*K)
    
    S_K_list = []
    for k in range(K):
        S_k = S_Y[k*p+np.array(range(p))[:, None], k*p+np.array(range(p))[None, :]]
        S_k = S_k - S_0
        S_K_list.append(S_k)
    
    S_hat_list = [S_0] + S_K_list
    S_pd0_list = []
    # closest PSD
    for k in range(K+1):
        S_pd0_list.append(nearestPD(S_hat_list[k]))
    
    lam_max_vec = [alpha_max(item) for item in S_pd0_list]
    lam1_max = lam_max_vec[0]
    lam2_max = max(lam_max_vec[1:])
    lam1_vec = np.logspace(np.log10(lam1_max*5e-1), np.log10(lam1_max), set_length)[::-1]
    lam2_vec = np.logspace(np.log10(lam2_max*5e-1), np.log10(lam2_max), set_length)[::-1]
    
    lam_product = itertools.product(lam1_vec, lam2_vec)
    lam_product_list = list(lam_product)
    
    Omega_grid_list = []
    for i in range(set_length):
        lam1_0 = lam_product_list[set_length*i][0]
        lam2_0 = lam_product_list[set_length*i][1]
        lam_vec = [lam1_0]+list(np.repeat(lam2_0, K))
        
        #initialize
        Omega_list = [cov.graph_lasso(S_pd0_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 
        
        A = np.zeros([p,p])
        for k in range(K):
            A = A + Omega_list[k]
        A_inv = alg.inv(A)
        
        likelihood_K = 0
        for k in range(K):
            likelihood_K = likelihood_K + np.log(alg.det(Omega_list[k+1])) - np.trace(np.matmul(S_Y[k*p+np.array(range(p))[:, None], k*p+np.array(range(p))[None, :]], Omega_list[k+1]))
        
        tr_OSOA = 0
        for m in range(K):
            for l in range(K):
                OSOA = np.matmul(np.matmul(np.matmul(Omega_list[m+1],S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]), Omega_list[l+1]), A_inv)
                tr_OSOA = tr_OSOA + np.trace(OSOA)
        
        likelihood = likelihood_K + np.log(alg.det(Omega_list[0])) - np.log(alg.det(A)) + tr_OSOA
        
        penalty = 0
        for k in range(K+1):
            penalty = penalty + lam_vec[k]*np.sum(np.abs(np.tril(Omega_list[k], -1) + np.triu(Omega_list[k], 1)))             
        
        pen_likelihood = likelihood - penalty 
    
        Omega_lam1_list = []     
        for j in range(set_length):
            lam2 = lam_product_list[set_length*i+j][1]
            lam_vec = [lam1_0] + list(np.repeat(lam2, K))
            pen_likelihood0 = 0
            while np.abs(pen_likelihood - pen_likelihood0) > 1e-4:       
                OSO = np.zeros([p,p])
                S_K_list = []
                for m in range(K):
                    SO = np.zeros([p,p])
                    OS = np.zeros([p,p])
                    for l in range(K):
                        S_ml = S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]
                        S_lm = S_Y[l*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
                        SO = SO + np.matmul(S_ml, Omega_list[l+1])
                        OS = OS + np.matmul(Omega_list[l+1], S_lm)
                        OSO = OSO + np.matmul(np.matmul(Omega_list[m+1],S_ml), Omega_list[l+1])
                    S_mm = S_Y[m*p+np.array(range(p))[:, None], m*p+np.array(range(p))[None, :]]
                    S_k = S_mm - np.matmul(SO, A_inv) - np.matmul(A_inv, OS)
                    S_K_list.append(S_k)
            
                S_0 = A_inv + np.matmul(np.matmul(A_inv,OSO), A_inv)
                S_K_list = [item + S_0 for item in S_K_list]
                S_pd_list = [S_0] + S_K_list
                
                #M
                Omega_list = [cov.graph_lasso(S_pd_list[k], lam_vec[k], verbose = False)[1] for k in range(K+1)] 
                
                # likelihood
                A = np.zeros([p,p])
                for k in range(K):
                    A = A + Omega_list[k]
                A_inv = alg.inv(A)
                    
                likelihood_K = 0
                for k in range(K):
                    likelihood_K = likelihood_K + np.log(alg.det(Omega_list[k+1])) - np.trace(np.matmul(S_Y[k*p+np.array(range(p))[:, None], k*p+np.array(range(p))[None, :]], Omega_list[k+1]))
                
                tr_OSOA = 0
                for m in range(K):
                    for l in range(K):
                        OSOA = np.matmul(np.matmul(np.matmul(Omega_list[m+1],S_Y[m*p+np.array(range(p))[:, None], l*p+np.array(range(p))[None, :]]), Omega_list[l+1]), A_inv)
                        tr_OSOA = tr_OSOA + np.trace(OSOA)
                
                likelihood = likelihood_K + np.log(alg.det(Omega_list[0])) - np.log(alg.det(A)) + tr_OSOA
                
                penalty = 0
                for k in range(K+1):
                    penalty = penalty + lam_vec[k]*np.sum(np.abs(np.tril(Omega_list[k], -1) + np.triu(Omega_list[k], 1)))             
                
                pen_likelihood0 = pen_likelihood        
                pen_likelihood = likelihood - penalty
                print(pen_likelihood)
            Omega_lam1_list.append(Omega_list)
            print([i, j])
        Omega_grid_list.append(Omega_lam1_list)
    return Omega_grid_list
#------------------------------------------- End defining private functions ----------------------------------------------------
    

