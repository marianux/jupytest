#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 07:53:40 2019

@author: mariano
"""

import scipy.signal as sig
import matplotlib as mpl
from splane import analyze_sys, pzmap, grpDelay, bodePlot
import numpy as np

#mpl.rcParams['figure.figsize'] = (15,15)
    
#####################
## Start of script ##
#####################
        
        
#aprox_name = 'Butterworth'
#aprox_name = 'Chebyshev1'
#aprox_name = 'Chebyshev2'
aprox_name = 'Bessel'
#aprox_name = 'Cauer'

ripple = [3, 3, 3] # dB \alpha_{max} <-- Sin parametrizar, lo dejo en Butterworth
#ripple = [1, 3, 6] # dB \alpha_{max}

attenuation = [40, 40, 40]  # dB \alpha_{min} <-- Sin parametrizar, att fija
#attenuation = [20, 40, 60]  # dB \alpha_{min}

#orders2analyze = [2, 2, 2] # <-- Sin parametrizar, orden fijo
orders2analyze = [2, 3, 4]

all_sys = []
filter_names = []

for (this_order, this_ripple, this_att) in zip(orders2analyze, ripple, attenuation):

    if aprox_name == 'Butterworth':
    
        z,p,k = sig.buttap(this_order)
        
        eps = np.sqrt( 10**(this_ripple/10) - 1 )
        num, den = sig.zpk2tf(z,p,k)
        num, den = sig.lp2lp(num, den, eps**(-1/this_order))
        
        z,p,k = sig.tf2zpk(num, den)
    
    elif aprox_name == 'Chebyshev1':
    
        z,p,k = sig.cheb1ap(this_order, this_ripple)
        
    elif aprox_name == 'Chebyshev2':
    
        z,p,k = sig.cheb2ap(this_order, this_ripple)
        
    elif aprox_name == 'Bessel':
        
        z,p,k = sig.besselap(this_order, norm='mag')
        
    elif aprox_name == 'Cauer':
       
        z,p,k = sig.ellipap(this_order, this_ripple, this_att)


    num, den = sig.zpk2tf(z,p,k)
    
    all_sys.append(sig.TransferFunction(num,den))
    filter_names.append(aprox_name + '_ord_' + str(this_order) + '_rip_' + str(this_ripple)+ '_att_' + str(this_att))


analyze_sys( all_sys, filter_names )