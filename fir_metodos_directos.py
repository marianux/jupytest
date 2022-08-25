#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy as np

'''
#################
# recordar que: #
#################
(de scipy firwin2 doc)

odd cant_coef, antisymmetric is False, type I filter is produced
even cant_coef, antisymmetric is False, type II filter is produced
odd cant_coef, antisymmetric is True, type III filter is produced
even cant_coef, antisymmetric is True, type IV filter is produced
Magnitude response of all but type I filters are subjects to following constraints:

type II – zero at the Nyquist frequency
type III – zero at zero and Nyquist frequencies
type IV – zero at zero frequency

'''
cant_coef = 51

#####################
## tipos de filtro ##
#####################

tipo_filtro = 'lp' # pasa bajo
#tipo_filtro = 'hp' # pasa alto
#tipo_filtro = 'bp' # pasa banda
#tipo_filtro = 'bs'  # elimina banda

# plantilla
ripple = 0.5 # dB
atenuacion = 40 # dB

if tipo_filtro == 'lp':
    # pasa bajo
    frecs = [0.0,  0.5,     0.8,          1.0]
    gains = [0,   -ripple, -atenuacion,   -atenuacion] # dB

elif tipo_filtro == 'hp':
    # pasa alto
    frecs = [0.0,         0.5,     1.0]
    gains = [-atenuacion, -ripple, 0.0] # dB

elif tipo_filtro == 'bp':
    # pasa banda
    frecs = [0.0,         0.4,     0.5, 0.6,     1.0]
    gains = [-atenuacion, -ripple, 0.0, -ripple, -atenuacion]

else:
    # elimina banda
    frecs = [0.0, 0.3,     0.4,        0.6,         0.7,     1.0]
    gains = [0.0, -ripple, -atenuacion, -atenuacion, -ripple, 0.0]
    
gains = 10**(np.array(gains)/20)
    
fs = 2.0
    
# algunas ventanas para evaluar
#win_name = 'boxcar'
win_name = 'hamming'
#win_name = 'blackmanharris'
#win_name = 'flattop'


# FIR design
num_win = sig.firwin2(cant_coef, frecs, gains , window=win_name )
num_firls = sig.firls(cant_coef, frecs, gains, fs=fs)
num_remez = sig.remez(cant_coef, frecs, gains[::2], fs=fs)

# coeficientes a_0 = 1; a_i = 0, i=1:cant_coef para los filtros FIR
den = 1

ww, hh_win = sig.freqz(num_win, den)
_,  hh_firls = sig.freqz(num_firls, den)
_,  hh_remez = sig.freqz(num_remez, den)
ww = ww / np.pi

plt.figure()

plt.plot(ww, 20 * np.log10(abs(hh_win)), label='win')
plt.plot(ww, 20 * np.log10(abs(hh_firls)), label='ls')
plt.plot(ww, 20 * np.log10(abs(hh_remez)), label='remez')

plt.plot(frecs, 20*np.log10(gains), 'rx', label='plantilla' )

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

plt.figure()

plt.plot(ww, np.unwrap(np.angle(hh_win)), label='win')
plt.plot(ww, np.unwrap(np.angle(hh_firls)), label='ls')
plt.plot(ww, np.unwrap(np.angle(hh_remez)), label='remez')

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [rad]')
plt.grid(which='both', axis='both')
plt.show()





