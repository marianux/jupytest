#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy as np

from pytc2.sistemas_lineales import plot_plantilla, group_delay


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

cant_coef = 21

#####################
## tipos de filtro ##
#####################

filter_type = 'lowpass'
# filter_type = 'highpass'
# filter_type = 'bandpass'
# filter_type = 'bandstop'


# plantilla
ripple = 0.5 # dB
attenuation = 40 # dB

if filter_type == 'lowpass':
    
    fpass = 0.25 # 
    fstop = 0.6 # Hz
    
    # pasa bajo
    frecs = [0.0,  fpass,     fstop,          1.0]
    gains = [0,   -ripple, -attenuation,   -attenuation] # dB

elif filter_type == 'highpass':
    # pasa alto
    frecs = [0.0,          fstop,        fpass,     1.0]
    gains = [-attenuation, -attenuation, -ripple,    0.0] # dB

elif filter_type == 'bandpass':
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

ww_rad, hh_win = sig.freqz(num_win, den)
_,  hh_firls = sig.freqz(num_firls, den)
_,  hh_remez = sig.freqz(num_remez, den)
ww = ww_rad / np.pi

plt.close('all')

plt.figure(1)

plt.plot(ww, 20 * np.log10(abs(hh_win)), label='win')
plt.plot(ww, 20 * np.log10(abs(hh_firls)), label='ls')
plt.plot(ww, 20 * np.log10(abs(hh_remez)), label='remez')

plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

plt.figure(2)

phase_win = np.unwrap(np.angle(hh_win))
phase_firls = np.unwrap(np.angle(hh_firls))
phase_remez = np.unwrap(np.angle(hh_remez))

plt.plot(ww, phase_win, label='win')
plt.plot(ww, phase_firls, label='ls')
plt.plot(ww, phase_remez, label='remez')

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [rad]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

gd_win = group_delay(ww_rad, phase_win)
gd_firls = group_delay(ww_rad, phase_firls)
gd_remez = group_delay(ww_rad, phase_remez)

plt.figure(3)

plt.plot(ww, gd_win, ':o', label='win')
plt.plot(ww, gd_firls, ':o', label='ls')
plt.plot(ww, gd_remez, ':o', label='remez')

plt.ylim((np.min((gd_win,gd_firls,gd_remez))-1, np.max((gd_win,gd_firls,gd_remez))+1))
plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Retardo [# muestras]')
plt.grid(which='both', axis='both')


axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()
