#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:44:03 2018

@author: monicawang76
"""
import sys
import os
sys.path.append(os.path.abspath('..')+'/TVGL')
import pickle

import numpy as np
from myfunctions import *
# from random import *
import random
# import snap
# import numpy.linalg as alg
from scipy.linalg import block_diag
#import sklearn.covariance as cov
#import TVGL as tvgl
#import myTVGL as mtvgl

#---------------------------------------- Generating basic structures ----------------------------------------------
random.seed(10)
np.random.seed(123)

p0 = 20
p1 = 20
d0 = 1
d1 = 1
ni = 50
n_change = 2
len_t = 11

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
    
#f = open("mydata.pkl", 'wb')
#pickle.dump(X_list, f)
#pickle.dump(A_list, f)
#f.close()
#-------------------------------------------------------------------------------------------------------------------------------------------------

