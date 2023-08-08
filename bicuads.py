#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

from pytc2.sistemas_lineales import tf2sos_analog, analyze_sys, pretty_print_SOS


# num
qn = -np.sqrt(2)/2
wn = 5
# den
qp = np.sqrt(2)/2
wp = 5

# kn = 1/wn**2 
# kp = 1/wp**2 

kn = 1 
kp = 1 

# coeficientes
# num = kn * np.array([1, 0])
# # Omega y Q
num = kn * np.array([1, wn/qn, wn**2]) 

# den = kp * np.array([1, 2, 2, 1])
# Omega y Q
den = kp * np.array([1, wp/qp, wp**2])

# tf_bicuad_sos = np.hstack((num,den)).reshape((1,6))
tf_bicuad_sos = tf2sos_analog( num, den )


# pretty_print_SOS(tf_bicuad_sos, mode='omegayq')
pretty_print_SOS(tf_bicuad_sos, mode='omegayq')
 
plt.close('all')

# analyze_sys(tf_bicuad_sos, 'mi_bicuad', same_figs=False)
analyze_sys([sig.TransferFunction(num,den)], 'mi_bicuad', same_figs=False)

# para editar la vista de la figura
# 
# aa = plt.gcf()
# bb = aa.get_axes()
# plt.sca(bb[0])
# plt.ylim([-40, 1])
# bb[0].get_legend().remove()
# bb[0].set_xlabel('Angular frequency [rad/sec]')
# bb[0].get_xticks()