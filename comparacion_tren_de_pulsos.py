# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:37:56 2020

@author: Torres Molina Emmanuel O.
"""

"""
Comparación de dos Filtros Normalizados Pasa-Bajos Butterworth y Chebyshev de
Orden 5, cuando filtran un "Tren de Pulsos" de pulsación angular < 1/5 [rad/seg]

Voy a Suponer un w = 1/8

"""

# Importo los Paquetes, Métodos a Utilizar:
from splane import bodePlot, pzmap, convert2SOS, analyze_sys, pretty_print_lti
from scipy.signal import TransferFunction as tfunction
from scipy.fftpack import fft
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

plt.close ( 'all' )


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Datos y Valores Dados:
# w_max = 1/16    #  Pulsación Angular     [rad/seg]
# f_max = w_max / (2*np.pi)  # Frecuencia Máxima
w_pulsos = 1/10
f_pulsos = w_pulsos / (2*np.pi)  # Frecuencia Máxima

# wp = w_max
# fp = f_max

# wp_prima = wp / wp

# Proceso de Muestreo:
#fN = 2.5 * f_max   # Frecuencia de Nyquist: fN >= 2*Fmáx
fN = 10 / (2*np.pi)   # Frecuencia de Nyquist: fN >= 2*Fmáx

Ts = 1 / fN  # Período de Sampling
Fs = 1/Ts  # Frecuencia de Sampling

N = 201 # Cantidad de Muestras.

t0 = 0
dt = Ts
tf = (N-1)*Ts

# Grilla de Sampleo Temporal
t = np.linspace (t0, tf, N) # array temporal de 100 muestras equispaciadas Ts.

# Onda Cuadrada
wave_square = sig.square (2 * np.pi * f_pulsos * t )

# Conformo el Tren de Pulsos de Amplitud 1V:
train_pulse = (wave_square >= 0)
train_pulse = train_pulse * 1

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Diseño los Filtros Normalizados de Orden 5 para Luego Compararlos:

orden_filtro = 5       # Orden de los Filtros.
eps = 1 

# ----------------------------------------------------------------------------

print ('\n\nFiltro Butterworth de Orden 5:\n')

# Filtro de Butterworth: Uso de los Métodos dentro del Paquete signal de Scipy.
z1, p1, k1 = sig.buttap (orden_filtro)

# Obtengo los Coeficientes de mi Transferencia.
NUM1, DEN1 = sig.zpk2tf (z1, p1, k1)

# Cálculo de wb:
wb = eps ** (-1/orden_filtro)

# Obtengo la Transferencia Normalizada
my_tf_bw = tfunction ( NUM1, DEN1 )
pretty_print_lti(my_tf_bw)

print ('\nTransferencia Normalizada Factorizada:', convert2SOS ( my_tf_bw ))

NUM1, DEN1 = sig.lp2lp ( NUM1, DEN1, wb )
my_tf_bw = tfunction ( NUM1, DEN1 )

# ----------------------------------------------------------------------------

print ('\n\nFiltro Chebyshev de Orden 5:\n')

# Filtro de Chebyshev: Uso de los Métodos dentro del Paquete signal de Scipy.
z2, p2, k2 = sig.cheb1ap (orden_filtro, eps)

# Obtengo los Coeficientes de mi Transferencia.
NUM2, DEN2 = sig.zpk2tf (z2, p2, k2)

# Obtengo la Transferencia Normalizada
my_tf_ch = tfunction ( NUM2, DEN2 )
pretty_print_lti(my_tf_ch)

print ('\nTransferencia Normalizada Factorizada:', convert2SOS ( my_tf_ch ))


# ----------------------------------------------------------------------------

print ('\n\nFiltro Chebyshev de Orden 5:\n')

# Filtro de Chebyshev: Uso de los Métodos dentro del Paquete signal de Scipy.
z3, p3, k3 = sig.besselap(orden_filtro, norm='mag')

# Obtengo los Coeficientes de mi Transferencia.
NUM3, DEN3 = sig.zpk2tf (z3, p3, k3)

# Obtengo la Transferencia Normalizada
my_tf_bessel = tfunction ( NUM3, DEN3 )
pretty_print_lti(my_tf_bessel)

print ('\nTransferencia Normalizada Factorizada:', convert2SOS ( my_tf_bessel ))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# Ploteo de las Señales, Respuestas, y Filtros:
my_tfs = [my_tf_bw , my_tf_ch, my_tf_bessel]

# # Respuesta de Módulo y Fase
# # Diagrama de Polos Y Ceros   
# for i in range ( len(my_tfs) ):
    
#     bodePlot ( my_tfs[i], 1)
#     plt.legend (['Filtro LP Butter Orden 5', 'Filtro LP Cheby Orden 5'])
#     pzmap ( my_tfs[i], 2 )
    
    
analyze_sys(my_tfs, ['Butter', 'Cheby', 'Bessel'])


tt, y1, x = sig.lsim2 ((NUM1, DEN1), train_pulse, t )
tt, y2, x = sig.lsim2 ((NUM2, DEN2), train_pulse, t )
tt, y3, x = sig.lsim2 ((NUM3, DEN3), train_pulse, t )

fig5 = plt.figure ()
fig5.suptitle ("Tren de Pulsos de Amplitud 1V y de 0.125 [rad/seg] Muestreado a Ts = 2.425 seg durante 240 seg")
plt.plot (t, train_pulse)
plt.grid()
plt.xlabel ('t[seg]')
plt.ylabel ('x[t] [V]')
# plt.xlim (0, 450)
plt.ylim (-0.1, 1.1)


fig6 = plt.figure ( )
fig6.suptitle ( 'Comparación entre las Salidas al Aplicar el Filtro Pasa-Bajos' )
plt.plot (t, train_pulse, 'g')
plt.plot (t, y1, 'r')
plt.plot(t, y2, 'b')
plt.plot(t, y3, 'k')
plt.legend ( ['Tren de Pulsos', 'Butter', 'Cheby', 'Bessel'] )
plt.grid()
plt.xlabel ('time [seg]')
plt.ylabel ('y(t) [V]')
# plt.xlim (0, 450)
plt.ylim (-0.1, 1.2)

"""
# Grilla de Sampleo Frecuencial
f0 = 0
df = Fs/N 
f_f = (N-1)*df
ff = np.linspace (f0, f_f, N)


X = np.abs ( fft (train_pulse)  ) 
Y1 = np.abs ( fft (y1)  )
Y2 = np.abs ( fft (y2)  )

fig7 = plt.figure ( )
fig7.suptitle ( 'Comparación de la FFT del Tren de Pulsos' )
plt.plot (ff[1:], X[1:], 'g')
plt.plot (ff[1:], Y1[1:], 'r' )
plt.plot (ff[1:], Y2[1:], 'b' )
plt.legend ( ['FFT del Tren de Pulsos', 'FFT del Filtro Butter', 'FFT del Filtro de Cheby'] )
plt.grid ( )
plt.xlabel ( 'Frecuency [Hz]' )
plt.ylabel ( 'Magnitude Response' )
plt.xlim (0.000995, 0.048)

"""