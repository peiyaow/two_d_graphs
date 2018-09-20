#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:56:03 2018

@author: peiyao
"""

import rpy2 
import rpy2.robjects as robjects
# import rpy2.robjects.packages as rpackages
robjects.r('''
        GetSigX = function(A, eigen.threshold = 0.01){
        	# this function is to make sure the matrix have lowest positive eigen value
        	# This will keep the same eigen vector but only threshold the eigen value to
        	# a presepecified eigen value
        	eig.A = eigen(A)$value
        	id = (eig.A < eigen.threshold)
        	eig.A[id] = eigen.threshold
        	result = eigen(A)$vectors %*% diag(eig.A) %*% t(eigen(A)$vectors)
        	return(result)	
        }
        
        MatrixMaxProj = function(M, r = 0.5, tol_value = 0.01, N = 100) {
        	#This is the projection to make the matrix positive definite
        	t = 0  
        	p = dim(M)[1]
        	Rt = GetSigX(M, 0.01)
        	Zt = matrix(rep(1/p^2, p^2), p, p)
        	while(t < N){ 
        		R0t = Rt - GetSigX(Rt - Zt, eigen.threshold = 0) # ex
        		Z0t = Zt - PB1.projection(Rt + Zt - M)     # ez
        		diff = max(max(abs(R0t)), max(abs(Z0t)))
        		cat(sep = "", "[",t,"]","dif=", round(diff, 3), "\n")
        		if(diff < tol_value){
        			break
        		}
        		Rt = Rt - r * (R0t - Z0t)/2
        		Zt = Zt - r * (R0t + Z0t)/2
        		t = t + 1
        	}
        	A = as.matrix(Rt)
        	return(A)
        }
        
        PB1.projection = function(A){
        	# This is to find matrix A.result where |A.result|_1 <=1
        	# min ||  A - A.result||_F
        	p = dim(A)[1]
        	a = as.vector(A)
        	T.mat = T.matrix(A * sign(A))
        	a1 = T.mat %*% (sign(a) * a)  
        	a.del = Delta.vec(a1)
        	a.del.cum = Cumulate.vec( a.del * (1:p^2) )
        	b = Cumulate.vec(a1)
        	if(b[p^2] <= 1){
        		x = a1    
        	} else {
        		id = (a.del.cum < 1)
        		K = sum(id) + 1
        		y = (b[K] - 1)/K
        		x = rep(0, p^2)
        		x[1:K] = a1[1:K] - y
        	}
        	B1 = sign(a) * solve(T.mat) %*% x
        	A.result = matrix(B1, p, p)
        	A.result = (A.result + t(A.result))/2
        	return(A.result)
        }
        
        T.matrix = function(A){
        	p = dim(A)[1]
        	a = as.vector(A)  
        	id = order(a, decreasing = TRUE)
        	T.mat = matrix(0, p^2, p^2)
        	mat.id = cbind(1:p^2, id)
        	T.mat[mat.id] = 1  
        	return(T.mat)
        }
        
        Delta.vec = function(a){
        	# calculate the different between consective entries
        	p = length(a)
        	a.result = rep(0, p)
        	a.result[p] = a[p]
        	a.result[1:(p - 1)] = a[1:(p - 1)] - a[2:p]
        	return(a.result)
        }
        
        Cumulate.vec = function(a){
        	# calculate the cumulate sum for a vector
        	p = length(a)
        	b = rep(0, p)
        	for(i in 1:p){
        		b[i] = sum(a[1:i])
        	}
        	return(b)
        }
        ''')

r_MatrixMaxProj = robjects.globalenv['MatrixMaxProj']

# import rpy2's package module
import rpy2.robjects.packages as rpackages

if not rpackages.isinstalled('QUIC'):    
    # import R's utility package
    utils = rpackages.importr('utils')
    
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=145) # select the first mirror in the list
    utils.install_packages("QUIC")

robjects.r('''
           library("QUIC")
           ''')

r_QUIC = robjects.r['QUIC']


