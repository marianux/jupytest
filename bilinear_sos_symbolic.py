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


#%% Resolución simbólica

s, z = sp.symbols('s z', complex=True)
k, fs, Q, Om = sp.symbols('k fs Q Om', real=True, positive=True)

Ts = 1/(s**2+s/Q+1)
fz = k * (z-1)/(z+1)

Tz = sp.collect(sp.simplify(sp.expand(Ts.subs(s, fz))), z)

display(Ts)
display(Tz)


#%% Parte numérica 


fs = 2 # Hz (Normalizamos a fs/2 = f_nyq)

# fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 


allQ = np.array([0.5, np.sqrt(2)/2, 5])
allfs = np.array([ 1, 2, 4])

this_q = np.sqrt(2)/2 # Butter
this_fs = allfs[0] # fs

plt.close('all')

for this_fs in allfs:
    
    for this_q in allQ[:-1]:

    
        k = 2 * this_fs
            
        kz2 = this_q * k**2  + this_q + k
        kz1 = -2 * this_q * k**2 + 2 * this_q
        kz0 = this_q * k**2  + this_q - k
        
        numz =  this_q * np.array([1, 2, 1])
        denz =  np.array([kz2, kz1, kz0])
        
        my_df = sig.TransferFunction(numz, denz, dt=1/fs)
        
        #filter_description='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs)
        pzmap(my_df, annotations = False,  fig_id=1)
        
        bodePlot(my_df, fig_id=2, digital = True, fs = this_fs)
        
    
    # el último le ponemos anotación para que quede lindo el gráfico
    this_q = allQ[-1]
    k = 2 * this_fs
        
    kz2 = this_q * k**2  + this_q + k
    kz1 = -2 * this_q * k**2 + 2 * this_q
    kz0 = this_q * k**2  + this_q - k
    
    numz =  this_q * np.array([1, 2, 1])
    denz =  np.array([kz2, kz1, kz0])
    
    my_df = sig.TransferFunction(numz, denz, dt=1/fs)
    
    pzmap(my_df, annotations = False, filter_description='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs), fig_id=1)

    bodePlot(my_df, fig_id=2, digital = True, fs = this_fs, filter_description ='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs))

    # bodePlot(my_df, fig_id=2, digital = True)
    # bodePlot(myFilter, fig_id='none', axes_hdl='none', filter_description = '', npoints = 1000, digital = False, fs = 2*np.pi ):
   


    
    