#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from pytc2.sistemas_lineales import analyze_sys, pretty_print_lti, tf2sos_analog, pretty_print_SOS


# de la plantilla pasaaltos llegamos a la plantilla pasabajo
alfa_max = 3 # dB
fo = 2000e3
f1 = 1600e3
f2 = 2500e3
fs1 = 1250e3
fs2 = 3200e3

Bbp = f2-f1
Q_bp = fo/Bbp

# normalizo respecto a fo
w1 = f1/fo
w2 = f2/fo
ws1 = fs1/fo
ws2 = fs2/fo
wo_bp = 1

# cuentas auxiliares

# epsilon cuadrado
ee = 10**(alfa_max/10)-1

Om1 = Q_bp * np.abs(ws1**2-wo_bp**2)/ws1
Om2 = Q_bp * np.abs(ws2**2-wo_bp**2)/ws2

for nn in range(2,5):
    # butter
    alfa_min_w1 = 10*np.log10(1 + ee * Om1**(2*nn) )
    alfa_min_w2 = 10*np.log10(1 + ee * Om2**(2*nn) )
    print( 'nn {:d} - {:f} [Hz]: {:f} dB - {:f} [Hz]: {:f} dB '.format(nn, w1, alfa_min_w1, w2, alfa_min_w2) )

    # cheby    
    # alfa_min_w1 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om1))**2 )
    # alfa_min_w2 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om2))**2 )
    # print( 'nn {:d} - {:f} [Hz]: {:f} dB - {:f} [Hz]: {:f} dB '.format(nn, f1, alfa_min_w1, f2, alfa_min_w2) )
   

# elijo un orden luego de iterar ...
nn = 3
alfa_min_w1 = 10*np.log10(1 + ee * Om1**(2*nn) )
alfa_min_w2 = 10*np.log10(1 + ee * Om2**(2*nn) )
ee = 1 #butter

print('\n\n')
print('----------------------------')
print('* Elegimos de la iteración *')
print('----------------------------')

print( 'n {:d} - {:f} [Hz]: {:f} dB - {:f} [Hz]: {:f} dB '.format(nn, f1, alfa_min_w1, f2, alfa_min_w2) )

# verificación 
z,p,k = sig.buttap(nn)
num_pb, den_pb = sig.zpk2tf(z,p,k)

print('\n\n')
print('--------------------------------')
print('Transferencia pasabajo prototipo')
print('--------------------------------')
pretty_print_lti(num_pb, den_pb)

# particiono en 2 SOS's para la implementación
sos_pb = tf2sos_analog(num_pb, den_pb)

# parametrizada
pretty_print_SOS(sos_pb, mode='omegayq')

# 
num_pbanda, den_pbanda = sig.lp2bp(num_pb, den_pb, bw = 1/Q_bp)

print('\n\n')
print('--------------------------------')
print('Transferencia pasabanda prototipo')
print('--------------------------------')
pretty_print_lti(num_pbanda, den_pbanda)

# particiono en 2 SOS's para la implementación
sos_pbanda = tf2sos_analog(num_pbanda, den_pbanda)

print('\n\n')
print('------------------')
print('Particiono en SOSs')
print('------------------')

# la visualizamos de algunas formas, la tradicional
pretty_print_SOS(sos_pbanda)

print('\n\n')
print('------------------------------------------------')
print('Particiono en SOSs parametrizados como nos gusta')
print('------------------------------------------------')

# o parametrizada
pretty_print_SOS(sos_pbanda, mode='omegayq')

plt.close('all')

analyze_sys( sos_pbanda )
