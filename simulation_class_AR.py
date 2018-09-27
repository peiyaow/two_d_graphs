import numpy as np
import sys
import os
#sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('/home/peiyao/GitHub/'))
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *
from two_d_graphs.MatrixMaxProj import *
from scipy import io
from statsmodels.stats.moment_helpers import cov2corr

p = 50
d = 3
n_change = 3
len_t = 201 # T
n = 20 # N
h = 5.848/np.cbrt(len_t-1)/2
sigma = 0.1
phi = 0.5

gG = getGraph(p, d)
S, A = gG
A_T = getGraphatT_Shuheng(S, A, n_change)[1]
A_T_list = [lam*A_T+(1-lam)*A for lam in np.linspace(0, 1, len_t)] # Omega
C_T_list = [getCov(item) for item in A_T_list] # Cov: 0+class time
#C_T_list = [alg.inv(item) for item in A_T_list] # Cov: 0+class time

gG0 = getGraph(p, d)
S0, A0 = gG0
C0 = getCov(A0)
#C0 = alg.inv(A0)

X = np.random.multivariate_normal(mean = np.zeros(p), cov = C0, size = n)
E_list = []
E0 = np.zeros([n,p])
for t in range(len_t):
    E = phi*E0 + np.random.multivariate_normal(mean = np.zeros(p), cov = C_T_list[t]*sigma**2, size = n)
    E0 = E
    E_list.append(E)
Y = X + np.array(E_list)

X_hat = Y.mean(0)
C0_hat = np.dot(np.transpose(X_hat - X_hat.mean(0)), X_hat - X_hat.mean(0))/n
X_mean = X.mean(0)
C0_X_hat = np.dot(np.transpose(X - X_mean), X - X_mean)/n

# Y
kernel_vec = np.array([np.exp(-np.square((m - 0)/(h*(len_t-1)))) for m in range(len_t)])
kernel_mtx = np.zeros([len_t, len_t])
for t_ix in range(len_t-1):
    kernel_mtx[t_ix, np.arange(t_ix+1, len_t)] = kernel_vec[np.arange(1, len_t-t_ix)]
kernel_mtx = kernel_mtx + np.transpose(kernel_mtx) + np.eye(len_t)

var_E = []
for i in range(n):
    var_i = []
    for t in range(len_t):
        var_s = []
        for s in range(len_t):
            var_s.append(kernel_mtx[t,s]*np.outer(Y[:,i,:][s,:], Y[:,i,:][s,:]))
        var_i.append(np.sum(np.array(var_s),0)/np.sum(kernel_mtx[t,:]))
    var_E.append(var_i)

alpha_1 = alpha_max(cov2corr(C0_hat))
alpha_0 = alpha_1*0.1
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
A0_hat_list = [cov.graph_lasso(cov2corr(C0_hat), alpha)[1] for alpha in alphas]

alpha_1 = alpha_max(np.array(cov2corr(C0)))
alpha_0 = alpha_1*0.1
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
A0_oracle_list = [cov.graph_lasso(np.array(cov2corr(C0)), alpha)[1] for alpha in alphas]

alpha_1 = alpha_max(cov2corr(C0_X_hat))
alpha_0 = alpha_1*0.1
alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), 50)
A0_X_list = [cov.graph_lasso(cov2corr(C0_X_hat), alpha)[1] for alpha in alphas]

P_list, R_list, P_X_list, R_X_list, P_hat_list, R_hat_list = [],[],[],[],[],[]
for i in range(50):
    P,R = getF1(A0, A0_oracle_list[i])[0:2]
    P_X,R_X = getF1(A0, A0_X_list[i])[0:2]
    P_hat,R_hat = getF1(A0, A0_hat_list[i])[0:2]
    P_list.append(P)
    R_list.append(R)
    P_X_list.append(P_X)
    R_X_list.append(R_X)
    P_hat_list.append(P_hat)
    R_hat_list.append(R_hat)

import matplotlib.pyplot as plt
plt.plot(P_list, R_list, P_X_list, R_X_list, P_hat_list, R_hat_list)
