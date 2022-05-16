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
alfa_max = 1
ws = 3.5

# cuentas auxiliares

# epsilon cuadrado
ee = 10**(alfa_max/10)-1

for nn in range(2,5):
    
    alfa_min_b = 10*np.log10(1 + ee * ws**(2*nn))
    

# elijo un orden luego de iterar ...
nn = 4
alfa_min_b = 10*np.log10(1 + ee * ws**(2*nn))
ee = 10**(alfa_max/10)-1

print( 'nn {:d} - eps^2: {:f} alfa_min {:f}'.format(nn, ee, alfa_min_b) )

# verificación MP
z,p,k = sig.buttap(nn)
num, den = sig.zpk2tf(z,p,k)

# Cualquier camino es válido, pero ojo con el signo de \omega_butter
num_mp_pb, den_mp_pb = sig.lp2lp(num, den, ee**(-1/2/nn))
# obtenemos la transferencia completa pasa-altos
num_mp_pa, den_mp_pa = sig.lp2hp(num_mp_pb, den_mp_pb)

# num_mp_pb, den_mp_pb = sig.lp2lp(num, den)
# # obtenemos la transferencia completa pasa-altos
# num_mp_pa, den_mp_pa = sig.lp2hp(num_mp_pb, den_mp_pb, ee**(1/2/nn))


print('\n\n')
print('--------------------------------')
print('Transferencia pasabajo prototipo')
print('--------------------------------')
tc2.pretty_print_lti(num_mp_pb, den_mp_pb)

# obtenemos la transferencia completa pasa-altos
print('\n\n')
print('----------------------------------')
print('Transferencia pasa-altos prototipo')
print('----------------------------------')
tc2.pretty_print_lti(num_mp_pa, den_mp_pa)

# particiono en 2 SOS's para la implementación
sos_pa = tc2.tf2sos_analog(num_mp_pa, den_mp_pa)

print('\n\n')
print('------------------')
print('Particiono en SOSs')
print('------------------')

# la visualizamos de algunas formas, la tradicional
tc2.pretty_print_SOS(sos_pa)

print('\n\n')
print('------------------------------------------------')
print('Particiono en SOSs parametrizados como nos gusta')
print('------------------------------------------------')

# o parametrizada
tc2.pretty_print_SOS(sos_pa, mode='omegayq')

plt.close('all')

tc2.analyze_sys( sos_pa )
