#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:43:34 2018

@author: MonicaW
"""
# nt:137 p:116
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *
from scipy import io

Y = io.loadmat('fMRI_AD.mat')['data0']
K = Y.shape[1]
p = Y[0,0].shape[1]
len_t = Y[0,0].shape[2]
n_vec = [Y[0,k].shape[0] for k in range(K)]
n = sum(n_vec)
h = 5.848/np.cbrt(len_t)

Y_list = [Y[0,k].transpose([0,2,1]) for k in range(K)] # ni by p by t -> each element: ni by time*p

test = Y_list[0]


