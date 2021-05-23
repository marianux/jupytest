#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:19:00 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import analyze_sys, pzmap, grpDelay, bodePlot

alfa_max = 1 # dB
alfa_min = 35 # dB
omega_p = 1 # norm omega
omega_s = 3.5 # norm omega

eps = np.sqrt(10**(alfa_max/10)-1)

print( 'eps = {:3.3f}  -  eps**2 = {:3.3f}'.format(eps, eps**2) )

for ii in range(1,6):
    print('alpha_{:d} = {:3.3f} dB'.format( ii, 10*np.log10(1 + eps**2 * omega_s**(2*ii)) ) ) 




z,p,k = sig.buttap(3)
num_lp, den_lp = sig.zpk2tf(z,p,k)
eps = np.sqrt(10**(0.1)-1)

num_lp_d, den_lp_d = sig.lp2lp(num_lp,den_lp, eps**(-1/3) )
num_hp_d, den_hp_d = sig.lp2hp(num_lp_d, den_lp_d)

num_hp_d, den_hp_d = sig.lp2hp(num_lp,den_lp, eps**(-1/3) )

#%matplotlib qt5
analyze_sys([sig.TransferFunction(num_lp, den_lp)], ['mp_norm'])
analyze_sys([sig.TransferFunction(num_lp_d,den_lp_d)], ['mp_desnorm'])
