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

from splane import analyze_sys, plot_plantilla

#####################
## Start of script ##
#####################

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)
        

#%% 

############################################
## Diseño a partir de un filtro analógico ##
############################################

# No tenemos control sobre las bandas de paso, transición y detenida (en el dominio digital) con este 
# tipo de metodología. Trabajamos en la plantilla analógica, escogemos una frecuencia de muestreo (fs)
# adecuada con la plantilla, y obtenemos el filtro digital.
# El proceso de diseño es iterativo hasta cumplir con la plantilla.

# Tipo de aproximación.
        
aprox_name = 'Butterworth'
# aprox_name = 'Chebyshev1'
# aprox_name = 'Chebyshev2'
# aprox_name = 'Cauer'

# Requerimientos de plantilla

# prototipo analógico

filter_type = 'lowpass'

f0 = 0.25 # normalizado respecto a nyq (fs/2)
w0 =  2*np.pi*f0  # 
# fc = w0/2/np.pi # Hz
ripple = 0.5
attenuation = 40
order2analyze = 3

fpass = f0 # 
fstop = 0.6 # 


# frecuencia de muestreo (solo será útil para el caso desnormalizado)
fs = 2
# fs = 2*np.pi

# Prewarp
# la función bilinear de scipy espera la fs como parámetro de ajuste.
# por eso le pasaremos fpw/2.
# fpw = 2*fs; # sin prewarp
# prewarp para que se iguale la transferencia en módulo y fase para w0.
fpw = w0/np.tan(w0/2/fs); 



all_sys = []
analog_filters = []
digital_filters = []

filter_names = []
analog_filters_names = []
digital_filters_names = []

if aprox_name == 'Butterworth':

    eps = np.sqrt(10**(ripple/10)-1)
    
    z,p,k = sig.buttap(order2analyze)

    # Desnormalizamos para cumplir con el ripple
    z, p, k = sig.lp2lp_zpk(z, p, k, wo=w0*eps**(-1/order2analyze) )

elif aprox_name == 'Chebyshev1':

    z,p,k = sig.cheb1ap(order2analyze, ripple)

    z, p, k = sig.lp2lp_zpk(z, p, k, wo=w0 )
    
elif aprox_name == 'Chebyshev2':

    z,p,k = sig.cheb2ap(order2analyze, attenuation)

    z, p, k = sig.lp2lp_zpk(z, p, k, wo= 2*np.pi*fstop )
    
elif aprox_name == 'Cauer':
   
    z,p,k = sig.ellipap(order2analyze, ripple, attenuation)

    z, p, k = sig.lp2lp_zpk(z, p, k, wo=w0 )

num, den = sig.zpk2tf(z,p,k)

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


plt.figure(1)
plt.cla()

npoints = 1000
w_nyq = 2*np.pi*fs/2

w, mag, _ = my_analog_filter.bode(npoints)
plt.plot(w/w_nyq, mag, label=my_analog_filter_desc)

w, mag, _ = my_digital_filter.bode(npoints)
plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plt.gca().set_xlim([0, 2])

plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)


