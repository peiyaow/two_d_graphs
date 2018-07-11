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
# import snap
import numpy.linalg as alg
from scipy.linalg import block_diag
import sklearn.covariance as cov

import networkx as nx
from scipy import sparse

import multiprocessing as mp

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
    
#    S_new = S_new + S_new.T
    
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
#         print('No edge detected on off diagonal, precision is zero')
        P = np.nan
    R = TandD/T
    
    if P+R:
        F1 = 2*P*R/(P+R)
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

def PD_array_simple(Theta_array_simple, A_list, class_ix):
    # Theta_array: class by time by alpha by p by p
    P_list, R_list = getPD(Theta_array_simple, A_list, class_ix)
    result = np.array([P_list, R_list]) # 2(P or R) by alpha
    return result

# set working directory as TVGL_result
def get_average_array(name):
    glasso_P_array = np.array([]).reshape(0,51)
    glasso_R_array = np.array([]).reshape(0,51)
    for i in range(50):
        path = "./glasso/glasso"+str(i)+".npy"    
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
        path = "./glasso/glasso"+str(i)+".npy"    
        path = "./"+name+"/"+name+str(i)+".npy" 
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

def Shuheng_method(X_list, ix_product, set_length, sigma, width = 5, knownMean = False):
    class_ix, time_ix = ix_product
    cov_last = None
    alpha_max_ct = alpha_max(genEmpCov(X_list[class_ix][time_ix].T))
    alpha_set = np.logspace(np.log10(alpha_max_ct*5e-2), np.log10(alpha_max_ct), set_length)
    result = []
    for alpha in alpha_set:
        emp_cov = genEmpCov_kernel(time_ix, sigma, width, X_list[class_ix], knownMean)
        ml_glasso = cov.graph_lasso(emp_cov, alpha, cov_init=cov_last, verbose = True) 
        cov_last = ml_glasso[0]
        result.append(ml_glasso[1])
    return result

# each element in sample_set is n by p len is total length of timestamp
# total width = 2*width + 1
def genEmpCov_kernel(t_query, sigma, width, sample_set, knownMean = False):
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
            S = S + K*np.dot((sample_set[j]).T, sample_set[j])/samplesPerStep
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
#------------------------------------------- End defining private functions ----------------------------------------------------
    

