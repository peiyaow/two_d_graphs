#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:53:19 2018

@author: monicawang76
"""
# for tvgl
X_array = np.array(X_list)
X_array = np.transpose(X_array, [1, 0, 2, 3]) # time by class by ni by p
X_array = np.reshape(X_array, (len_t, len_class*ni, p)) # time by ni*len_class by p
result = mtvgl.myTVGL0(X_array, ni, 0.1, 0.1, 1, True, h)

# for mytvgl0
X_array = np.array(X_list) # class by time by ni by p
X_array = np.reshape(X_array, (len_class, len_t*ni, p)) # time by ni*len_class by p
result0 = mtvgl.myTVGL(X_array, ni, 0.1, .1, 1, True, h)
#X_concat_class_list = []
#for class_ix in range(len_class):
#    X_concat = np.concatenate(X_list[class_ix])
#    X_concat_class_list.append(X_concat)