#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5: Filtros digitales

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from IPython.display import display


#%% Resolución simbólica

z = sp.symbols('z', complex=True)
D = sp.symbols('D', real=True, positive=True)
k = sp.symbols('k', integer=True, positive=True)

# promediador clásico
Tma_clasico = 1/D * sp.Sum(z**(-k), (k, 0, D-1))

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
# con D múltiplo de 2**NN, dado que el escalado por D sería simplemente 
# una rotación a nivel de bits de NN veces a la derecha, y su implementación
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

display(Tdc_removal_2)

# Ahora con 4 moving average

Tdc_removal_4 = z**-(2*D-2) - Tma**4

# emprolijamos la expresion
num, den = Tdc_removal_4.as_numer_denom()
num = (sp.expand(num/(D**4*z**(4*D+4)))).powsimp()
den = (sp.expand(den/(D**4*z**(4*D+4)))).powsimp()

Tdc_removal_4 = num/den

display(Tdc_removal_4)

#%% Parte numérica 

fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2

NN = 2**13
w_rad  = np.append(np.logspace(-2, 0.8, NN//4), np.logspace(0.9, 1.6, NN//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, NN//2, endpoint=True) ) / nyq_frec * np.pi

def transf_s_2ba( T_s ):
    
    num, den = sp.fraction(T_s)
    
    bb = np.array(num.as_poly(z**-1).all_coeffs(), dtype=float)
    aa = np.array(den.as_poly(z**-1).all_coeffs(), dtype=float)
    
    return( bb,aa )


def group_delay( freq, phase):
    
    dphase = -np.diff(np.unwrap(phase, period = 19/10* np.pi ))
    # dphase = -np.diff(phase)
    # # corregir saltos de fase
    # bAux = dphase > np.pi
    # dphase[bAux] = np.amin( np.hstack([dphase[bAux], dphase[bAux]-np.pi]), axis = 1 )
    groupDelay = dphase/np.diff(freq)
    
    return(np.append(groupDelay, groupDelay[-1]))
    


def Sym_freq_response(HH, zz, ww):
    
    # w = sp.Symbol('w', real=True)
    # HH_jw = HH.subs({z:1*sp.exp_polar(sp.I*w)})
    
    H_numeric = sp.lambdify(zz, HH, modules=['numpy'])
    
    z_vals = np.exp(1j * ww)  # Rango de frecuencias angulares
    
    # Evalúa la función numérica en el rango de frecuencias angulares
    magnitude_response = np.abs(H_numeric(z_vals))
    phase_response = np.angle(H_numeric(z_vals))  # La fase se devuelve en grados

    return((magnitude_response, phase_response))


def plt_freq_resp(title, magnitude_response, phase_response, w_rad, fs = 2):
    
    ww = w_rad / np.pi * fs/2

    # Grafica la respuesta en frecuencia de módulo
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(ww, 20 * np.log10(magnitude_response))
    
    plt.title('Respuesta en Frecuencia de Módulo: ' + title)
    plt.xlabel('Frecuencia Angular (w)')
    plt.ylabel('|H(jw)| (dB)')

    plt.axis([0, 100, -60, 5 ]);
    
    gd = group_delay( w_rad, phase_response)
    mgd = np.nanmedian(gd)
    
    # Grafica la respuesta en frecuencia de retardo
    plt.subplot(2, 1, 2)
    plt.plot(ww, gd)
    plt.title('Respuesta de retardo')
    plt.xlabel('Frecuencia Angular (w)')
    plt.ylabel('Retardo (#)')
    plt.axis([0, 100, mgd-5, mgd+5 ]);
    
    plt.tight_layout()
    plt.show()
    

def blackman_tukey(x,  M = None):    
    
    N = len(x)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    return Px;


# Del análisis simbólico
DD = [8, 16, 64]
UU = 20
  

# Grafica la respuesta en frecuencia de módulo

plt.figure(1)
plt.clf()

ww = w_rad / np.pi * fs / 2

for ddd in DD:
    
    # Cálculo de demoras para mov. avg
    demora_ma = int(UU*(ddd-1))

    Tma_ddd = Tma.subs({z:z**UU, D:ddd})

    mod_Tma_ddd, pha_Tma_ddd = Sym_freq_response(Tma_ddd, z, w_rad )

    plt.plot(ww, 20 * np.log10(mod_Tma_ddd), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_ma) )

plt.title('Respuesta en Frecuencia de Módulo: RL-MovAvgRec-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 500, -80, 1 ]);

plt.figure(2)
plt.clf()

ww = w_rad / np.pi * fs / 2

for ddd in DD:
    
    # Cálculo de demoras para mov. avg
    demora_ma = int(UU*(ddd-1))

    Tma_c_ddd = Tma_clasico.subs({z:z**UU, D:ddd})
    
    Tma_c_ddd = Tma_c_ddd.doit()

    mod_Tma_c_ddd, pha_Tma_c_ddd = Sym_freq_response(Tma_c_ddd, z, w_rad )

    plt.plot(ww, 20 * np.log10(mod_Tma_c_ddd), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_ma) )

    # como num-den TF
    num, den = transf_s_2ba(Tma_c_ddd)


plt.title('Respuesta en Frecuencia de Módulo: RL-MovAvgClasico-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 500, -80, 1 ]);

plt.figure(3)
plt.clf()


for ddd in DD:
    
    # Cálculo de demoras para 2 mov. avg
    demora_rl2 = int(UU*(ddd-1))

    Tdcr_2 = Tdc_removal_2.subs({z:z**UU, D:ddd})

    mod_Tdcr_2, pha_Tdcr_2 = Sym_freq_response(Tdcr_2, z, w_rad )

    # plt_freq_resp('FIR-RL-2MA-D{:d}-OverS:{:d}'.format(DD, UU), mod_Tdcr_2, pha_Tdcr_2, w_rad, fs = fs)
    plt.plot(ww, 20 * np.log10(mod_Tdcr_2), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_rl2) )

plt.title('Respuesta en Frecuencia de Módulo: RL-2MA-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 100, -1, 0.5 ]);

plt.figure(4)
plt.clf()

for ddd in DD:

    # Cálculo de demoras para 4 mov. avg
    demora_rl4 = int(2*UU*(ddd-1))
    
    Tdcr_4 = Tdc_removal_4.subs({z:z**UU, D:ddd})
    
    mod_Tdcr_4, pha_Tdcr_4 = Sym_freq_response(Tdcr_4, z, w_rad )
    
    # plt_freq_resp('FIR-RL-4MA-D{:d}-OverS:{:d}'.format(DD, UU), mod_Tdcr_4, pha_Tdcr_4, w_rad, fs = fs)
    
    # plt.subplot(2, 1, 1)
    plt.plot(ww, 20 * np.log10(mod_Tdcr_4), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_rl4) )

plt.title('Respuesta en Frecuencia de Módulo: RL-4MA-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 100, -1, 0.5 ]);

plt.tight_layout()
plt.show()

