#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:55:28 2020

@author: mariano
"""

import numpy as np
from splane import bodePlot
from scipy.signal import TransferFunction
import matplotlib.pyplot as plt

# Cantidad de iteraciones o experimentos
NN = 500
# Tolerancia de los componentes
tol = 5

# Q y \omega_0 proyectados
QQ = 10
W0 = 1

# Valores de los componentes 
CC = 1
RR = 1
RB = (2-1/QQ)*RR

# Valores de los componentes para cada iteración:
# Cada valor es muestreado independientemente de una distribución uniforme,
# limitada por la tolerancia impuesta.
all_C = np.random.uniform(CC * (100-tol/2)/100 , CC * (100+tol/2)/100, size=NN )
all_R = np.random.uniform(RR * (100-tol/2)/100 , RR * (100+tol/2)/100, size=NN )
all_RB = np.random.uniform(RB * (100-tol/2)/100 , RB * (100+tol/2)/100, size=NN )

plt.close('all')

# analizaremos cada iteración resultante
for (this_C, this_R, this_RB) in zip( all_C, all_R, all_RB):

    this_KK = 1 + this_RB/this_R
    this_QQ = 1/(3-this_KK)
    this_w0 = 1/this_R/this_C
    
    my_tf = TransferFunction( [this_KK * (this_w0**2)], [1, this_w0/this_QQ, this_w0**2] )
    
    _, axes_hdl = bodePlot(my_tf, 1)
    
# finalmente ploteamos también la transferencia con los valores esperados
# sin incertidumbre alguna sobre sus valores.
KK = 1 + RB/RR
QQ = 1/(3-KK)
WW0 = 1/RR/CC

my_tf = TransferFunction( [KK * (WW0**2)], [1, WW0/QQ, WW0**2] )

w, mag, phase = my_tf.bode(n=300)

(mag_ax_hdl, phase_ax_hdl) = axes_hdl

plt.sca(mag_ax_hdl)
plt.semilogx(w, mag, '-r', linewidth=3 )    # Bode magnitude plot
    
plt.sca(phase_ax_hdl)
plt.semilogx(w, phase, '-r', linewidth=3)    # Bode phase plot


# Ahora vamos a hacer un estudio estadístico de los parámetros Q y \omega_0
# calculo los valores de los parámetros para TODAS las iteraciones
all_KK = 1 + all_RB/all_R
all_QQ = 1/(3-all_KK)
all_w0 = 1/all_R/all_C

plt.figure(2)
plt.hist( 20*np.log10(all_KK), 20 )
plt.title('Ganancia K para cada experimento')

plt.figure(3)
plt.hist( all_QQ, 20 )
plt.title('Q para cada experimento')

plt.figure(4)
plt.hist( all_w0, 20 )
plt.title('$\omega_0$ para cada experimento')



    

    
