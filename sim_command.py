#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:00:58 2018

@author: MonicaW
"""
import os
import numpy as np

nsim = 50
np.random.seed(10)
seeds = np.floor(1e4*np.random.uniform(size = [2,nsim])).astype(int)
for i in range(nsim):
#    system_string = "python2 myfunctions.py" + ' ' +  str(seeds[0,i]) + ' ' +str(seeds[1,i])
#    os.system("python2 test.py" + ' ' +  str(seeds[0,i]) + ' ' +str(seeds[1,i]))
    mystring = "python Theta_glasso_parallel.py" + ' ' +  str(seeds[0,i]) + ' ' + str(seeds[1,i]) + ' ' + str(i)
    os.system('sbatch -o glasso.out -t 00-10:00 -n 10 --mem-per-cpu 4G -N 1-1 --wrap="' + mystring + '"')

    
    
    
    



