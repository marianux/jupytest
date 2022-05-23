#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import scipy.signal as sig
import splane as tc2
import matplotlib.pyplot as plt


# de la plantilla pasaaltos llegamos a la plantilla pasabajo
alfa_max = 0.5 # dB
fo = 22e3
f1 = 17e3
f2 = 36e3

wo_bp = fo * 2 * np.pi
Q_bp = 5
w1 = f1 * 2 * np.pi/wo_bp
w2 = f2 * 2 * np.pi/wo_bp
wo_bp = 1

# cuentas auxiliares

# epsilon cuadrado
ee = 10**(alfa_max/10)-1

Om1 = Q_bp * np.abs(w1**2-wo_bp**2)/w1
Om2 = Q_bp * np.abs(w2**2-wo_bp**2)/w2

for nn in range(2,5):
    alfa_min_w1 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om1))**2 )
    alfa_min_w2 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om2))**2 )
    print( 'nn {:d} - {:f} [Hz]: {:f} dB - {:f} [Hz]: {:f} dB '.format(nn, f1, alfa_min_w1, f2, alfa_min_w2) )
   

# elijo un orden luego de iterar ...
nn = 3
alfa_min_w1 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om1))**2 )
alfa_min_w2 = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(Om2))**2 )
ee = 10**(alfa_max/10)-1

print('\n\n')
print('----------------------------')
print('* Elegimos de la iteración *')
print('----------------------------')

print( 'n {:d} - {:f} [Hz]: {:f} dB - {:f} [Hz]: {:f} dB '.format(nn, f1, alfa_min_w1, f2, alfa_min_w2) )

# verificación Cheby1
z,p,k = sig.cheb1ap(nn, alfa_max)
num_pb, den_pb = sig.zpk2tf(z,p,k)

print('\n\n')
print('--------------------------------')
print('Transferencia pasabajo prototipo')
print('--------------------------------')
tc2.pretty_print_lti(num_pb, den_pb)

# particiono en 2 SOS's para la implementación
sos_pb = tc2.tf2sos_analog(num_pb, den_pb)

# parametrizada
tc2.pretty_print_SOS(sos_pb, mode='omegayq')

# 
num_pbanda, den_pbanda = sig.lp2bp(num_pb, den_pb, bw = 1/Q_bp)

print('\n\n')
print('--------------------------------')
print('Transferencia pasabanda prototipo')
print('--------------------------------')
tc2.pretty_print_lti(num_pbanda, den_pbanda)

# particiono en 2 SOS's para la implementación
sos_pbanda = tc2.tf2sos_analog(num_pbanda, den_pbanda)

print('\n\n')
print('------------------')
print('Particiono en SOSs')
print('------------------')

# la visualizamos de algunas formas, la tradicional
tc2.pretty_print_SOS(sos_pbanda)

print('\n\n')
print('------------------------------------------------')
print('Particiono en SOSs parametrizados como nos gusta')
print('------------------------------------------------')

# o parametrizada
tc2.pretty_print_SOS(sos_pbanda, mode='omegayq')

plt.close('all')

tc2.analyze_sys( sos_pbanda )
