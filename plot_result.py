#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:41:43 2018

@author: monicawang76
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    

f = open("Theta_glasso.pkl", 'rb')
Theta_glasso_list = pickle.load(f)
f.close()

Theta_glasso_array = np.array(Theta_glasso_list) # 44 class by time
Theta_glasso_array = np.reshape(Theta_glasso_array, (4, 11, 51, 100, 100))

f = open("Theta_Shuheng.pkl", 'rb')
Theta_Shuheng_list = pickle.load(f)
f.close()
Theta_Shuheng_array = np.array(Theta_Shuheng_list)
Theta_Shuheng_array = np.transpose(Theta_Shuheng_array, [0, 2, 1, 3, 4])


f = open("Theta_mymethod.pkl", 'rb')
Theta_mymethod_list = pickle.load(f)
f.close()
Theta_mymethod_array = np.array(Theta_mymethod_list)
Theta_mymethod_array = np.reshape(Theta_mymethod_array, (51, 11, 11, 4, 100 ,100)) # alpha beta t class p p
Theta_mymethod_array = np.transpose(Theta_mymethod_array, [1, 3, 2, 0, 4, 5]) 
Theta_mymethod_array.shape
# beta, class, time, alpha, row, col

with open("mydata.pkl", 'rb') as f:
    X_list = pickle.load(f) 
    A_list = pickle.load(f)
f.close()

#array = np.random.randint(1,100,100)
# reshape the array to a 2D form

#array
#array = np.reshape(array, (4,25))
# reshape the array back to the 1D form



#---------------------------------------- Calculating P and R -------------------------------------------------
class_ix = 3

P_glasso_list, R_glasso_list = getPD(Theta_glasso_array, A_list, class_ix)
P_SH_list, R_SH_list = getPD(Theta_Shuheng_array, A_list, class_ix)

P_mymethod_list_list = []
R_mymethod_list_list = []
for i in range(11):
   P_mymethod_list, R_mymethod_list = getPD(Theta_mymethod_array[i], A_list, class_ix)
   P_mymethod_list_list.append(P_mymethod_list)
   R_mymethod_list_list.append(R_mymethod_list)
#--------------------------------------------------------------------------------------------------------------

#plt.plot(R_glasso_list, P_glasso_list, R_SH_list, P_SH_list)
#plt.legend(['glasso', 'Shuheng'])
#plt.show()

# ------------------------------------ plot to find the best beta ---------------------------------------------
for i in range(11):
    plt.plot(R_mymethod_list_list[i], P_mymethod_list_list[i])
plt.legend(range(11), loc=3, ncol = 3)    
plt.show()
#--------------------------------------------------------------------------------------------------------------

#------------------------------------- plot overall results ---------------------------------------------------
plt.plot(R_glasso_list, P_glasso_list, R_SH_list, P_SH_list, R_mymethod_list_list[4], P_mymethod_list_list[4])
plt.legend(['glasso', 'Shuheng', 'mymethod'], loc = 3)
plt.show()
#--------------------------------------------------------------------------------------------------------------










