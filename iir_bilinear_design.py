#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:57:54 2018

@author: mllamedo

Script para ejemplificar el diseño, análisis y uso de filtros IIR

"""

import numpy as np
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from splane import analyze_sys

#####################
## Start of script ##
#####################

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)
        

#%% 

############################################
## Diseño a partir de un filtro analógico ##
############################################

# Tipo de aproximación.
        
aprox_name = 'Butterworth'
#aprox_name = 'Chebyshev1'
#aprox_name = 'Chebyshev2'
#aprox_name = 'Cauer'

# Requerimientos de plantilla

# prototipo analógico

w0 =  0.5 # r/s normalizado respecto a nyq (fs/2)
# fc = 10e3/2/np.pi # Hz
fc = w0/2/np.pi # Hz
ripple = 0.5
attenuation = 40
order2analyze = 3

# frecuencia de muestreo (solo será útil para el caso desnormalizado)
# fs = 5*fc
fs = 2*np.pi

# Prewarp
# la función bilinear de scipy espera la fs como parámetro de ajuste.
# por eso le pasaremos fpw/2.
# fpw = 2*fs; # sin prewarp
# prewarp para que se iguale la transferencia en módulo y fase para w0.
fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 



all_sys = []
analog_filters = []
digital_filters = []

filter_names = []
analog_filters_names = []
digital_filters_names = []

if aprox_name == 'Butterworth':

    z,p,k = sig.buttap(order2analyze)

elif aprox_name == 'Chebyshev1':

    z,p,k = sig.cheb1ap(order2analyze, ripple)
    
elif aprox_name == 'Chebyshev2':

    z,p,k = sig.cheb2ap(order2analyze, ripple)
    
elif aprox_name == 'Cauer':
   
    z,p,k = sig.ellipap(order2analyze, ripple, attenuation)

num, den = sig.zpk2tf(z,p,k)

# Desnormalizamos para w0 (normalizada por w_nyq)
num, den = sig.lp2lp(num, den, w0*(2*np.pi*fs/2))

my_analog_filter = sig.TransferFunction(num,den)
my_analog_filter_desc = aprox_name + '_ord_' + str(order2analyze) + '_analog'

all_sys.append(my_analog_filter)
filter_names.append(my_analog_filter_desc)

analog_filters.append(my_analog_filter)
analog_filters_names.append(my_analog_filter_desc)

# Transformamos el filtro analógico mediante la transformada bilineal


numz, denz = sig.bilinear(num, den, fpw/2)

my_digital_filter = sig.TransferFunction(numz, denz, dt=1/fs)
my_digital_filter_desc = aprox_name + '_ord_' + str(order2analyze) + '_digital'

all_sys.append(my_digital_filter)
filter_names.append(my_digital_filter_desc)

plt.close('all')

# Analizamos y comparamos los filtros
analyze_sys( all_sys, filter_names, digital = True, fs = fs )


