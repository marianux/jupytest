#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import scipy.signal as sig
from splane import analyze_sys
import matplotlib.pyplot as plt


alfa_max = 1
ws = 2

# cuentas auxiliares

# epsilon cuadrado
ee = 10**(alfa_max/10)-1

for nn in range(2,5):
    
    alfa_min_b = 10*np.log10(1 + ee * ws**(2*nn))
    alfa_min_c = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(ws))**2 )
    
    print( 'nn {:d} - alfa_min_butter {:f} - alfa_min_cheby {:f}'.format(nn, alfa_min_b, alfa_min_c) )


# elijo un orden luego de iterar ...
nn = 2

# verificación MP
z,p,k = sig.buttap(nn)

num, den = sig.zpk2tf(z,p,k)
# aplico la renormalización a \omega_butter
num_mp, den_mp = sig.lp2lp(num, den, ee**(-1/2/nn))

# verificación Cheby
z,p,k = sig.cheb1ap(nn, alfa_max)
num_cheb, den_cheb = sig.zpk2tf(z,p,k)

# análisis de respuesta en frecuencia
all_sys = [sig.TransferFunction(num_mp, den_mp), 
           sig.TransferFunction(num_cheb, den_cheb)]

filter_names = ['MP_' + str(nn) + '_ripp_' + str(alfa_max) + 'dB',
                'Cheb_' + str(nn) + '_ripp_' + str(alfa_max) + 'dB']

plt.close('all')

analyze_sys( all_sys, filter_names)



