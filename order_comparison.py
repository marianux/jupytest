#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:57:54 2018

@author: mllamedo

Script para comparar distintos órdenes de aproximación en las functiones
de aproximación implementadas en scipy.signal

"""

import scipy.signal as sig
from splane import analyze_sys
import matplotlib as mpl

#####################
## Start of script ##
#####################

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)
        
# Tipo de aproximación.
        
aprox_name = 'Butterworth'
#aprox_name = 'Chebyshev1'
#aprox_name = 'Chebyshev2'
#aprox_name = 'Bessel'
#aprox_name = 'Cauer'

# Requerimientos de plantilla

ripple = 0.5
attenuation = 40
orders2analyze = range(2,7)

all_sys = []
filter_names = []

for ii in orders2analyze:

    if aprox_name == 'Butterworth':
    
        z,p,k = sig.buttap(ii)
    
    elif aprox_name == 'Chebyshev1':
    
        z,p,k = sig.cheb1ap(ii, ripple)
        
    elif aprox_name == 'Chebyshev2':
    
        z,p,k = sig.cheb2ap(ii, ripple)
        
    elif aprox_name == 'Bessel':
        
        z,p,k = sig.besselap(ii, norm='mag')
        
    elif aprox_name == 'Cauer':
       
        z,p,k = sig.ellipap(ii, ripple, attenuation)


    num, den = sig.zpk2tf(z,p,k)
    
    all_sys.append(sig.TransferFunction(num,den))
    filter_names.append(aprox_name + '_ord_' + str(ii))


analyze_sys( all_sys, filter_names )


