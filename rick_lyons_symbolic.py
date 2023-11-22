#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5: Filtros digitales

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
# from pytc2.sistemas_lineales import plot_plantilla, simplify_n_monic
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


#%% Resolución simbólica

z = sp.symbols('z', complex=True)
D = sp.symbols('D', real=True, positive=True)

# moving average
Tma = 1/D * (1-z**(-D))/(1-z**(-1))

# delay line of (D-1)/2
Tdl =  z**(-(D-1)/2)

num, den = (Tdl - Tma).as_numer_denom()

num = (sp.expand(num/(D*z**(D+1)))).powsimp()
den = (sp.expand(den/(D*z**(D+1)))).powsimp()

Tdc_removal = num/den

display(Tdc_removal)

# Según Rick Lyons, este sistema sería muy bueno para implementarse
# con D múltiplo de 2**N, dado que el escalado por D sería simplemente 
# una rotación a nivel de bits de N veces a la derecha, y su implementación
# no necesitaría de multiplicaciones. Sin embargo esta elección impone un 
# retardo no entero. Por esta razón se opta por poner dos (incluso cuatro) 
# sistemas idénticos en cascada.

# Probamos primero con 2 moving average

Tdc_removal_2 = z**-(D-1) - Tma**2

# emprolijamos la expresion a mano
num, den = Tdc_removal_2.as_numer_denom()
num = (sp.expand(num/(D**2*z**(2*D+2))).powsimp())
den = (sp.expand(den/(D**2*z**(2*D+2))).powsimp())


Tdc_removal_2 = num/den

def transf_s_2ba( T_s ):
    
    num, den = sp.fraction(T_s)
    
    bb = np.array(num.as_poly(z**-1).all_coeffs(), dtype=float)
    aa = np.array(den.as_poly(z**-1).all_coeffs(), dtype=float)
    
    return( (bb,aa) )

DD = 64
UU = 20
    
bb2_16, aa2_16 = transf_s_2ba( Tdc_removal_2.subs({z:z**UU, D:DD}) )

# display(Tdc_removal_2)

# Ahora con 4 moving average

Tdc_removal_4 = z**-(2*D-2) - Tma**4

# emprolijamos la expresion
num, den = Tdc_removal_4.as_numer_denom()
num = (sp.expand(num/(D**4*z**(4*D+4)))).powsimp()
den = (sp.expand(den/(D**4*z**(4*D+4)))).powsimp()

Tdc_removal_4 = num/den

bb4_16, aa4_16 = transf_s_2ba( Tdc_removal_4.subs({z:z**UU, D:DD}) )

# display(Tdc_removal_4)

#%% Parte numérica 

# Cargar el módulo Cythonizado
from recursive_fir_filter import filter_sequence

def one_MA_stage( double[:] xx, DD, UU):
    
    nn = len(vec)
    cdef double[:] yy
    cdef int i
    for i in range(nn):
        aux = aux + (1- aux)*vec[i]
    return aux
    
    
    return(yy)




fs = 1000 # Hz (Normalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2


# fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 

## Rick Lyons ECG filter

# demora_rl = int(uu*(dd-1))
demora2_rl = (len(bb2_16)-1)/2
demora2_rl = (len(bb4_16)-1)/2


N = 4000
w_rad  = np.append(np.logspace(-2, 0.8, N//4), np.logspace(0.9, 1.6, N//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, N//2, endpoint=True) ) / nyq_frec * np.pi

z,p,k = sig.tf2zpk(bb2_16, aa2_16)

sos_rl2 = sig.tf2sos(bb2_16, aa2_16, pairing='nearest')
sos_rl4 = sig.tf2sos(bb4_16, aa4_16, pairing='nearest')

_, hh2_rl = sig.sosfreqz(sos_rl2, w_rad)
_, hh4_rl = sig.sosfreqz(sos_rl4, w_rad)

w = w_rad / np.pi * nyq_frec

plt.close('all')

plt.figure(1)
plt.clf()

plt.plot(w, 20 * np.log10(abs(hh2_rl)), label='FIR-RL-2-D{:d} orden:{:d}'.format(2, DD))
plt.plot(w, 20 * np.log10(abs(hh4_rl)), label='FIR-RL-2-D{:d} orden:{:d}'.format(4, DD))
# plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.axis([0, 100, -60, 5 ]);
plt.legend()

plt.grid()

