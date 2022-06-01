#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import matplotlib.pyplot as plt
import numpy as np
from splane import tf2sos_analog, analyze_sys, pretty_print_SOS


# num
qn = -np.sqrt(2)/2
wn = 4
# den
qp = np.sqrt(2)/2
wp = 1

# kn = 1/wn**2 
# kp = 1/wp**2 

kn = 1 
kp = 1 

# coeficientes
num = kn * np.array([1, 0, wn**2])
# # Omega y Q
# num = kn * np.array([1, wn/qn, wn**2]) 

den = kp * np.array([1, wp/qp, wp**2])

tf_bicuad_sos = np.hstack((num,den)).reshape((1,6))

pretty_print_SOS(tf_bicuad_sos, mode='omegayq')
 
plt.close('all')

analyze_sys(tf_bicuad_sos, 'mi_bicuad',same_figs=False)

# para editar la vista de la figura
# 
# aa = plt.gcf()
# bb = aa.get_axes()
# plt.sca(bb[0])
# plt.ylim([-40, 1])
# bb[0].get_legend().remove()
# bb[0].set_xlabel('Angular frequency [rad/sec]')
# bb[0].get_xticks()