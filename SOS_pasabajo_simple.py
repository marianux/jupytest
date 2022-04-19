#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from splane import analyze_sys, pretty_print_bicuad_omegayq


# ejemplo simple
wo = 1
qq = np.sqrt(2)/2 

num = np.array([wo**2]) 
den = np.array([1, wo/qq, wo**2])

pretty_print_bicuad_omegayq(num,den)

mi_sos = sig.TransferFunction(num,den)
    
plt.close('all')
analyze_sys(mi_sos, 'mi SOS')


# # parametrización de wo o Q.
# all_sos = []
# all_values = [0.5, 1,5,10]

# for ii in all_values:
    
#     wo = ii
#     qq = np.sqrt(2)/2 
    
#     num = np.array([wo**2]) 
#     den = np.array([1, wo/qq, wo**2])
    
#     pretty_print_bicuad_omegayq(num,den)

#     mi_sos = sig.TransferFunction(num,den)

#     all_sos += [mi_sos]
    
# # plt.close('all')
# analyze_sys(all_sos, sys_name=all_values)

