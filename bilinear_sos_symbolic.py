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


def plot_response(my_df, this_q, this_fs):

    pzmap(my_df, annotations = False,  fig_id=1)

    plt.figure(2)

    w_rad, mag, phase = my_df.bode(npoints)
    
    ww = w_rad*2/this_fs/2/np.pi
    
    plt.plot(ww, mag, label='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs))
    
    plt.title('Bilenear demo')
    plt.xlabel('Frecuencia normalizada a Nyq [#]')
    plt.ylabel('Amplitud [dB]')
    plt.grid(which='both', axis='both')
    
    plt.figure(3)
    
    plt.plot(ww, phase, label='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs) )
    
    plt.title('Bilenear demo')
    plt.xlabel('Frequencia normalizada')
    plt.ylabel('Fase [grados]')
    plt.grid(which='both', axis='both')
    plt.show()
            
    plt.figure(4)
    
    # ojo al escalar Omega y luego calcular la derivada.
    gd = group_delay(w_rad, phase)
    
    plt.plot(ww, gd, label='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs))
    
    plt.title('Bilenear demo')
    plt.xlabel('Frequencia normalizada')
    plt.ylabel('Retardo [# muestras]')
    plt.grid(which='both', axis='both')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.show()
    

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
# analyze_sys(all_sys, all_sys_desc)

# Desnormalizado en Hz
# analyze_sys(all_sys, all_sys_desc, xaxis="freq")

# Normalizado individual de cada sistema
# analyze_sys(all_sys, all_sys_desc, xaxis="norm")

# Normalizado respecto a fs (Hz)
analyze_sys(all_sys, all_sys_desc, xaxis="norm", fs=allfs[-1]/2)



    
    