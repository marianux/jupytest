#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:57:54 2018

@author: mllamedo

Script para comparar distintos órdenes de aproximación en las functiones
de aproximación implementadas en scipy.signal

"""

import scipy.signal as sig
import matplotlib as mpl
from splane import analyze_sys


#####################
## Start of script ##
#####################

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)

# Requerimientos de plantilla


ripple = 3
attenuation = 40
order2analyze = 2

all_sys = []
filter_names = []

# Butter
z,p,k = sig.buttap(order2analyze)

num, den = sig.zpk2tf(z,p,k)

all_sys.append(sig.TransferFunction(num,den))

filter_names.append('Butter_ord_'+str(order2analyze))

# Chebyshev

z,p,k = sig.cheb1ap(order2analyze, ripple)

num, den = sig.zpk2tf(z,p,k)

all_sys.append(sig.TransferFunction(num,den))

filter_names.append('Cheby_ord_'+str(order2analyze))
    
# Bessel
    
z,p,k = sig.besselap(order2analyze, norm='mag')

num, den = sig.zpk2tf(z,p,k)

all_sys.append(sig.TransferFunction(num,den))

filter_names.append('Bessel_ord_'+str(order2analyze))

# Cauer
    
z,p,k = sig.ellipap(order2analyze, ripple, attenuation)

num, den = sig.zpk2tf(z,p,k)

all_sys.append(sig.TransferFunction(num,den))

filter_names.append('Cauer_ord_'+str(order2analyze))

# Analize and compare filters
analyze_sys( all_sys, filter_names )


