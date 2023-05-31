#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import scipy.signal as sig
from pytc2.sistemas_lineales import analyze_sys, tf2sos_analog, pretty_print_SOS, pretty_print_bicuad_omegayq
import matplotlib.pyplot as plt




alfa_max = 0.4
ws = 3

# cuentas auxiliares

# epsilon cuadrado
ee = 10**(alfa_max/10)-1

for nn in range(2,9):
    
    alfa_min_b = 10*np.log10(1 + ee * ws**(2*nn))
    alfa_min_c = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(ws))**2 )
    
    print( 'nn {:d} - alfa_min_butter {:f} - alfa_min_cheby {:f}'.format(nn, alfa_min_b, alfa_min_c) )
    # print( 'nn {:d} - alfa_min_butter {:f} '.format(nn, alfa_min_b) )


# elijo un orden luego de iterar ...
nn = 6

# verificación MP
z,p,k = sig.buttap(nn)

num, den = sig.zpk2tf(z,p,k)

# aplico la renormalización a \omega_butter
num_mp, den_mp = sig.lp2lp(num, den, ee**(-1/2/nn))

# num_hp, den_hp = sig.lp2hp(num_mp, den_mp)

# # sos_mp = tf2sos_analog(num_mp, den_mp)
# sos_hp = tf2sos_analog(num_hp, den_hp)

# verificación Cheby
z,p,k = sig.cheb1ap(nn, alfa_max)
num_cheb, den_cheb = sig.zpk2tf(z,p,k)

num_hp, den_hp = sig.lp2hp(num_cheb, den_cheb)

# sos_mp = tf2sos_analog(num_mp, den_mp)
sos_hp = tf2sos_analog(num_hp, den_hp)

# # análisis de respuesta en frecuencia
# all_sys = [sig.TransferFunction(num_mp, den_mp), 
            # sig.TransferFunction(num_cheb, den_cheb)]

# filter_names = ['MP_' + str(nn) + '_ripp_' + str(alfa_max) + 'dB',
#                 'Cheb_' + str(nn) + '_ripp_' + str(alfa_max) + 'dB']
# analyze_sys( all_sys, filter_names)

plt.close('all')

# analyze_sys( sig.TransferFunction(num_hp, den_hp))


analyze_sys( sos_hp)

# all_sys = [sig.TransferFunction(num_mp, den_mp), 
#             sig.TransferFunction(num_hp, den_hp)]

# filter_names = ['MP_PB' + str(nn) + '_ripp_' + str(alfa_max) + 'dB',
#                 'MP_PA' + str(nn) + '_ripp_' + str(alfa_max) + 'dB']

# analyze_sys( sig.TransferFunction(num_mp, den_mp) )

# pretty_print_bicuad_omegayq(num_mp, den_mp)

pretty_print_SOS(sos_hp, mode='omegayq')

# analyze_sys( all_sys)

