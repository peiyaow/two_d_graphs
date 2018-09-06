#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:59:49 2018

@author: MonicaW
"""

import numpy as np
from two_d_graphs.myfunctions import *
from two_d_graphs.posdef import *

from scipy import io

Y = io.loadmat('fMRI_AD.mat')
Y['data'][0,0].transpose([0, 2, 1]).reshape([]) # ni by p by t # each element: ni by time*p
