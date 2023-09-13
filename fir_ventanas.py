#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño por el método de ventanas.
Aprovechamiento del conocimiento de la transferencia para usar
los tipos de filtro FIR y las restricciones que imponen sobre
las transferencias.

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

type I – Sin restricciones.
type II – zero at the Nyquist frequency (pasabajos)
type III – zero at zero and Nyquist frequencies (pasabandas, Hilbert)
type IV – zero at zero frequency (pasaaltos, diferenciadores)

'''

cant_coef = 501
antisymmetric = False

# Por si quiero forzar un tipo de FIR

tipo_fir = 1 # Sin restricciones.
# tipo_fir = 2 # pasabajos
# tipo_fir = 3 # pasabandas, Hilbert
# tipo_fir = 4 # pasaaltos, diferenciadores

if( (tipo_fir == 2 or tipo_fir == 4) and cant_coef%2 == 1 ):
    
    # fuerzo par
    cant_coef += 1

if( (tipo_fir == 1 or tipo_fir == 3) and cant_coef%2 == 0 ):
    
    # fuerzo impar
    cant_coef += 1

if( (tipo_fir == 1 or tipo_fir == 2) and antisymmetric ):
   
    antisymmetric = False

if( (tipo_fir == 3 or tipo_fir == 4) and (not antisymmetric) ):
   
    antisymmetric = True

#####################
## tipos de filtro ##
#####################

filter_type = 'arbitrary'
# filter_type = 'lowpass'
# filter_type = 'highpass'
# filter_type = 'bandpass'
# filter_type = 'bandstop'


# plantilla
ripple = 0.5 # dB
attenuation = 40 # dB

if filter_type == 'arbitrary':
    
    # transferencia arbitraria
    frecs = [0.0,        0.2,     0.3,     0.4,      0.5,    0.6,   1.0 ]
    gains = [-np.inf,   -30,      -15,   -ripple,   -ripple, -50,  -np.inf] # dB

elif filter_type == 'lowpass':
    
    fpass = 0.25 # 
    fstop = 0.6 # 
    
    # pasa bajo
    frecs = [0.0,  fpass,     fstop,          1.0]
    gains = [0,   -ripple, -attenuation,   -np.inf] # dB

elif filter_type == 'highpass':

    fstop = 0.25 # 
    fpass = 0.6 # Hz

    # pasa alto
    frecs = [0.0,          fstop,        fpass,     1.0]
    gains = [-np.inf, -attenuation, -ripple,    0.0] # dB

elif filter_type == 'bandpass':
    
    fs1 = 0.25 # 
    fp1 = 0.5 # Hz
    fp2 = 0.55 # 
    fs2 = 0.8 # Hz
    
    fstop = [fs1,fs2] # 
    fpass = [fp1,fp2] # Hz
    
    # pasa banda
    frecs = [0.0,          fs1,          fp1,      fp2,     fs2,         1.0]
    gains = [-np.inf, -attenuation, -ripple, -ripple, -attenuation, -np.inf]

else:
    
    fp1 = 0.25 # Hz
    fs1 = 0.5 # 
    fs2 = 0.55 # Hz
    fp2 = 0.8 # 

    fstop = [fs1,fs2] # 
    fpass = [fp1,fp2] # Hz
    
    # elimina banda
    frecs = [0.0,      fp1,           fs1,          fs2,           fp2,      1.0]
    gains = [-ripple, -ripple,      -attenuation, -attenuation,  -ripple,  -ripple]
    
gains = 10**(np.array(gains)/20)
    
# fs = 1.0/np.pi
fs = 2.0

# algunas ventanas para evaluar
#win_name = 'boxcar'
#win_name = 'hamming'
win_name = 'blackmanharris'
#win_name = 'flattop'


# FIR design
num = sig.firwin2(cant_coef, frecs, gains , window=win_name, antisymmetric=antisymmetric  )
den = 1.0

# Una alternativa para utilizar el filtro diseñado:

########################
# OJO con la fase !!! ##
########################

digital_filter = sig.TransferFunction( num, den, dt=1/fs)

ww, module, phase = digital_filter.bode()
# devuelve ww en radianes. 2pi = Nyq
ww = ww / 2 / np.pi

plt.close('all')

plt.figure(1)

plt.plot(ww, module, label=filter_type+ '-' + win_name)

if filter_type == 'arbitrary':
    plt.plot(frecs, 20*np.log10(gains), 'rx', label='plantilla' )
else:
    plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)


plt.title('FIR designed by window method')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

plt.figure(2)

plt.plot(ww, phase, label=filter_type+ '-' + win_name )

plt.title('FIR designed by window method')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [grados]')
plt.grid(which='both', axis='both')
plt.show()


# Y otra ...

wrad, hh = sig.freqz(num, den)
ww = wrad / np.pi


plt.figure(3)

plt.plot(ww, 20 * np.log10(abs(hh)), label=filter_type+ '-' + win_name)

if filter_type == 'arbitrary':
    plt.plot(frecs, 20*np.log10(gains), 'rx', label='plantilla' )
else:
    plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)

plt.title('FIR designed by window method')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

plt.figure(4)

phase = np.unwrap(np.angle(hh))

plt.plot(ww, phase, label=filter_type+ '-' + win_name)

plt.title('FIR designed by window method')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [rad]')
plt.grid(which='both', axis='both')
plt.show()

plt.figure(5)


# ojo al escalar Omega y luego calcular la derivada.
gd_win = group_delay(wrad, phase)

plt.plot(ww, gd_win, ':o', label=filter_type+ '-' + win_name)

plt.ylim((np.min(gd_win)-1, np.max(gd_win)+1))
plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Retardo [# muestras]')
plt.grid(which='both', axis='both')


axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()




