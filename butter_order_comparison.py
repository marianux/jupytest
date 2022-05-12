#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:57:54 2018

@author: mllamedo

Script para comparar distintos órdenes de aproximación en las functiones de máxima planicidad (Butterworth) implementadas en scipy.signal.

"""
import numpy as np
import scipy.signal as sig
from splane import analyze_sys
import matplotlib as mpl
import matplotlib.pyplot as plt

#####################
## Start of script ##
#####################

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)
        
# Tipo de aproximación.
        
aprox_name = 'Butterworth'

# Requerimientos de plantilla

ripple = [1, 3, 6] # dB \alpha_{max} <-- Sin parametrizar, lo dejo en Butterworth

# orders2analyze = [2, 2, 2] # <-- Sin parametrizar, orden fijo
orders2analyze = [2, 2, 2]


all_sys = []
filter_names = []

for (this_order, this_ripple) in zip(orders2analyze, ripple):


    z,p,k = sig.buttap(this_order)
    
    eps = np.sqrt( 10**(this_ripple/10) - 1 )
    num, den = sig.zpk2tf(z,p,k)
    num, den = sig.lp2lp(num, den, eps**(-1/this_order))
    
    z,p,k = sig.tf2zpk(num, den)

    num, den = sig.zpk2tf(z,p,k)
    
    all_sys.append(sig.TransferFunction(num,den))
    filter_names.append('Butt_' + str(this_order) + '_ripp_' + str(this_ripple) + 'dB' )


plt.close('all')

# analizamos Todos los sistemas juntos para su comparación.
analyze_sys( all_sys, filter_names, annotations = False  )


