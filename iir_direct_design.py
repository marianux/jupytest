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

sys.path.append('/home/mariano/scripts/analog filters/python') 

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
        
aprox_name = 'butter'
#aprox_name = 'cheby1'
#aprox_name = 'cheby2'
#aprox_name = 'ellip'

# Por qué no hay bessel ?
#aprox_name = 'bessel'

# Requerimientos de plantilla

# filter_type = 'lowpass'
# filter_type = 'highpass'
# filter_type = 'bandpass'
filter_type = 'bandstop'


# plantillas normalizadas a Nyquist y en dB

if filter_type == 'lowpass':

    fpass = 0.25 # 
    ripple = 0.5 # dB
    fstop = 0.6 # Hz
    attenuation = 40 # dB

elif filter_type == 'highpass':

    fpass = 0.6 
    ripple = 0.5 # dB
    fstop = 0.25
    attenuation = 40 # dB

elif filter_type == 'bandpass':

    fpass = np.array( [0.4, 0.6] ) 
    ripple = 0.5 # dB
    fstop = np.array( [0.25, 0.75] ) 
    attenuation = 40 # dB
    
else:

    # bandstop
    fpass = np.array( [0.25, 0.75] ) 
    ripple = 0.5 # dB
    fstop = np.array( [0.4, 0.6] ) 
    attenuation = 40 # dB

# Variables 

all_sys = []
analog_filters = []
digital_filters = []

filter_names = []
analog_filters_names = []
digital_filters_names = []


#%% 

# Cálculo del filtro

# frecuencia de muestreo
fs = 2

if aprox_name == 'butter':

    order, wcutof = sig.buttord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.buttord( fpass, fstop, ripple, attenuation, analog=False)

elif aprox_name == 'cheby1':

    order, wcutof = sig.cheb1ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb1ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'cheby2':

    order, wcutof = sig.cheb2ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb2ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'ellip':
   
    order, wcutof = sig.ellipord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.ellipord( fpass, fstop, ripple, attenuation, analog=False)


# Diseño del filtro analógico

num, den = sig.iirfilter(order, wcutof, rp=ripple, rs=attenuation, btype=filter_type, analog=True, ftype=aprox_name)

my_analog_filter = sig.TransferFunction(num,den)
my_analog_filter_desc = aprox_name + '_ord_' + str(order) + '_analog'

all_sys.append(my_analog_filter)
filter_names.append(my_analog_filter_desc)

# Diseño del filtro digital

numz, denz = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, btype=filter_type, analog=False, ftype=aprox_name)

my_digital_filter = sig.TransferFunction(numz, denz, dt=1/fs)
my_digital_filter_desc = aprox_name + '_ord_' + str(orderz) + '_digital'

all_sys.append(my_digital_filter)
filter_names.append(my_digital_filter_desc)


# Plantilla de diseño

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

xmin, xmax, ymin, ymax = plt.axis()


# banda de paso digital
plt.fill([xmin, xmin, fs/2, fs/2],   [ymin, ymax, ymax, ymin], 'g', alpha= 0.2, lw=1, label = 'bw digital') # pass

if filter_type == 'lowpass':

    fstop_start = fstop
    fstop_end = xmax
    
    fpass_start = xmin
    fpass_end   = fpass

    plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
    plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass

elif filter_type == 'highpass':

    fstop_start = xmin
    fstop_end = fstop 
    
    fpass_start = fpass
    fpass_end   = xmax

    plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
    plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass


elif filter_type == 'bandpass':

    fstop_start = xmin
    fstop_end = fstop[0]
    
    fpass_start = fpass[0]
    fpass_end   = fpass[1]
    
    fstop2_start = fstop[1]
    fstop2_end =  xmax
    
    plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
    plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass
    plt.fill( [fstop2_start, fstop2_end,   fstop2_end, fstop2_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k') # stop
    
else:

    fpass_start = xmin
    fpass_end   = fpass[0]

    fstop_start = fstop[0]
    fstop_end = fstop[1]
    
    fpass2_start = fpass[1]
    fpass2_end   = xmax
        
    plt.fill([fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # pass
    plt.fill([fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k') # stop
    plt.fill([fpass2_start, fpass2_start, fpass2_end, fpass2_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass



plt.axis([xmin, xmax, np.max([ymin, -100]), np.max([ymax, 5])])

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

# Analizamos y comparamos los filtros
#analyze_sys( all_sys, filter_names )
