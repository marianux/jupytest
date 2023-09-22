#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5: Filtros digitales

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
from splane import pzmap, bodePlot
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from pytc2.sistemas_lineales import group_delay, analyze_sys


#%% Resolución simbólica

s, z = sp.symbols('s z', complex=True)
k, fs, Q, Om = sp.symbols('k fs Q Om', real=True, positive=True)

Ts = 1/(s**2+s/Q+1)
fz = k * (z-1)/(z+1)

Tz = sp.collect(sp.simplify(sp.expand(Ts.subs(s, fz))), z)

display(Ts)
display(Tz)

# display(Tz.subs(k, 2*fs))


#%% Parte numérica 


fs = 2 # Hz (Normalizamos a fs/2 = f_nyq)

# fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 


# allQ = np.array([0.5, np.sqrt(2)/2, 5])
# allfs = np.array([ 2, 4, 100])
allfs = np.array([ 2, 10, 100 ])
allQ = np.array([np.sqrt(2)/2])

all_sys = []
all_sys_desc = []

plt.close('all')

for this_fs in allfs:
    
    for this_q in allQ:

    
        k = 2 * this_fs
            
        a2 = this_q * k**2  + this_q + k
        a1 = -2 * this_q * k**2 + 2 * this_q
        a0 = this_q * k**2  + this_q - k
        
        numz =  this_q * np.array([1, 2, 1])
        denz =  np.array([a2, a1, a0])
        
        my_df = sig.TransferFunction(numz, denz, dt=1/this_fs)
        
        all_sys += [my_df]
        all_sys_desc += ['Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs)]
       

#%% probar diferentes perspectivas:
    
plt.close("all")
# Desnormalizado en radianes (default)
analyze_sys(all_sys, all_sys_desc)

# Desnormalizado en Hz
# analyze_sys(all_sys, all_sys_desc, xaxis="freq")

# Normalizado individual de cada sistema
# analyze_sys(all_sys, all_sys_desc, xaxis="norm")

# Normalizado respecto a fs (Hz)
# analyze_sys(all_sys, all_sys_desc, xaxis="norm", fs=allfs[-1]/2)



    
    