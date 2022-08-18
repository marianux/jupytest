#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:56:51 2021

En este script se simula un conversor analógico digital (ADC) mediante el 
re-muestreo y cuantización de una señal más densamente muestreada, simulando una señal continua en tiempo y amplitud. En el script se analiza el efecto del ALIAS, producto del muesrtreo, y el ruido de cuantización producido por la cuantización.
El experimento permite hacer una pre-distorsión de la "señal analógica" simulando un "piso de ruido", luego se analiza cómo afecta el mismo a la cuantización del ADC. También se puede analizar si la predistorsión está correlada (mediante una señal chirp) o incorrelada (ruido Gaussiano) respecto a la senoidal de prueba.

@author: mariano
"""

#%% Configuración e inicio de la simulación

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import neurokit2 as nk



# Datos generales de la simulación
fs = 500.0 # frecuencia de muestreo (Hz)
t_exp = 10 # segundos
N = int(t_exp * fs)   # cantidad de muestras

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 10
N_os = N*over_sampling

# Datos del ADC
B = 8 # bits
Vf = 2 # +/- Volts
q = 2*Vf/2**B # Volts

# proporción del rango del ADC "usado"
kl = 0.15

# datos del ruido analógico
# kn = 1/10 # poporción de la potencia del ruido de cuantización del ADC
# pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)
pot_ruido = -80 # dB
pot_ruido = 10**(pot_ruido/10) * N_os/2

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral


#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
tt_os = np.linspace(0, (N-1)*ts, N_os)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N-1)*df*over_sampling, N_os)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

# senoidal 
# analog_sig = np.sin( 2*np.pi*np.round(N*1/10)*df*tt_os )

# chirp
analog_sig = sig.chirp(tt_os, f0=1, t1=t_exp, f1=500 )

# ruido blanco
# analog_sig = np.random.normal(0, 1, size=N_os)

# ECG simulado
# analog_sig = nk.ecg_simulate(sampling_rate = fs*over_sampling, length = N_os, noise = 0.0, method="ecgsyn")

# conformación de la señal analógica: ubicar la señal entre +/- Vf utilizando 
# la proporción kl de ese rango.
# analog_sig = analog_sig * 2*Vf*kl/(np.max(analog_sig) - np.min(analog_sig))

# normalización en potencia.
analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

# Generación de la señal de interferencia
# incorrelada
nn = np.random.normal(0, np.sqrt(pot_ruido), size=N_os)

# muy correlada
# nn = sig.chirp(tt_os, 2*df, (N-1)*ts, fs/2)
# nn = nn / np.sqrt(np.var(nn)) * np.sqrt(pot_ruido)

# construimos la señal de entrada al ADC
sr = analog_sig + nn
# sr = analog_sig 

# El ADC limita los valores por encima y por debajo de +/-Vf.
sr = np.clip(sr, a_min = -Vf, a_max = Vf) 

# muestreo la señal analógica 1 cada OS muestras
# como el submuestreo es muy sencillo, over_sampling es un entero.
# sr = sig.filtfilt(num_remez, den_remez, sr)
sr = sr[::over_sampling]

# cuantizo la señal muestreada
srq = q * np.round(sr/q)

# ruido de cuantización
nq = srq - sr

nall = srq - analog_sig[::over_sampling]

# srq = sr
# nq = nn[::over_sampling]

#%% Presentación gráfica de los resultados
plt.close('all')

plt.figure(1)
plt.plot(tt, srq, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(tt, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
plt.plot( np.array([ tt[0], tt[-1] ]), np.array( [[ Vf, -Vf ], [ Vf, -Vf ]]), color='orange', ls='dashed',  label= '$ V_f $' )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr, axis = 0 )
ft_Srq = 1/N*np.fft.fft( srq, axis = 0 )
ft_As = 1/N_os*np.fft.fft( analog_sig, axis = 0)
ft_Nq = 1/N*np.fft.fft( nq, axis = 0 )
ft_Nn = 1/N_os*np.fft.fft( nn, axis = 0 )
bfrec = ff <= fs/2


As_mean = np.mean(np.abs(ft_As)**2)
Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec],            10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff[bfrec],            10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )
plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
plt.plot( ff[bfrec],            10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))



plt.figure(3)

ft_As2 = 1/N*np.fft.fft( analog_sig[::over_sampling], axis = 0)
ft_Nall = 1/N*np.fft.fft( nall, axis = 0 )


plt.plot( ff[bfrec], 10* np.log10(np.abs(ft_As2[bfrec])**2 / (np.abs(ft_Nall[bfrec])**2) ) ) 
plt.title('SNR de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('SNR [dB]')
plt.xlabel('Frecuencia [Hz]')



plt.figure(4)
bins = 10
plt.hist(nq, bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))


plt.figure(5)

fff, ttt, Sxx = sig.spectrogram(analog_sig, fs=fs*over_sampling, axis=0, nperseg=int(fs*over_sampling/5) )

plt.pcolormesh(ttt, fff, np.abs(Sxx), cmap='gray_r', shading='gouraud')


fff, ttt, Sxx = sig.spectrogram(srq, fs=fs, axis=0, nperseg=int(fs/5) )

plt.pcolormesh(ttt, fff, np.abs(Sxx), cmap='gray_r', shading='gouraud')

plt.plot( np.array([ tt[0], tt[-1] ]), np.array( [ fs/2, fs/2 ]), color='orange', ls='dashed',  label= 'Nyq' )

plt.ylim([ 0, fs] )

plt.title('Espectrograma')

plt.xlabel('t (sec)')

plt.ylabel('Frequency (Hz)')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.grid()

