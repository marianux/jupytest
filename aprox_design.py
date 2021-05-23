#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  23 07:53:40 2021

@author: mariano

Este script ejemplifica el uso de las funciones de diseño de filtros analógicos que se incluyen en SciPy. Para usarlo deberás completar los datos de tu plantilla normalizada de atenuación y el tipo de aproximación (Butter, Cheby o Bessel). Otra opción es forzar el orden del polinomio aproximante para alguna de dichas funciones. El script obtiene los coeficientes de la función transferencia, y grafica la respuesta en frecuencia del filtro diseñado (módulo, fase y retardo).

"""

import scipy.signal as sig
import matplotlib as mpl
from splane import analyze_sys, pretty_print_lti
import numpy as np

#mpl.rcParams['figure.figsize'] = (15,15)

###############
## funciones ##
###############

def besselord( omega_p, omega_s, alfa_max, alfa_min, omega_d, max_pc_delay ):

    min_order = 0

    # partimos de suponer que al menos será el orden de un Butter o más grande
    start_order, _ = sig.buttord(omega_p, omega_s, alfa_max, alfa_min+alfa_max, analog=True)
    
    # de forma iterativa, intentamos cumplimentar la plantilla
    for ii in range(start_order, 20):
        
        z,p,k = sig.besselap(ii, norm='delay')
        
        this_lti = sig.ZerosPolesGain(z, p, k).to_tf()
        
        _, mm, pp = sig.bode(this_lti, w=[0, 0.0001, omega_p, omega_d, omega_d + 0.0001 ])

        # attenuation in omega_p, i.e. bandpass end
        this_ripple = -mm[2];
        
        # relative delay 
        this_delay = 1 - np.abs((pp[4] - pp[3])/(pp[1] - pp[0]))
        
        if this_ripple <= alfa_max and this_delay <= max_pc_delay:
            break
    
    min_order = ii

    return min_order    
    
    
#####################
## Start of script ##
#####################

# Plantilla de módulo
#####################
alfa_max = 1 # dB
alfa_min = 18 # dB
omega_p = 1 # norm omega
omega_s = 3 # norm omega

# Plantilla de demora (solo para Bessel)
#########################################
# nota: también se considera la plantilla de módulo en Bessel, aunque no la 
# omega_s y alfa_min.

omega_d = 2.5 # norm omega. Pulsación a la cual se compara la demora relativa.
max_pc_delay = 0.1 # error relativo de demora, en omega_d, respecto al centro de la banda de paso.


# Aproximación
        
# aprox_name = 'Butterworth'
# aprox_name = 'Chebyshev1' # equiripple banda de paso
# aprox_name = 'Chebyshev2' # equiripple banda de rechazo
# aprox_name = 'Bessel'
aprox_name = 'Cauer' # o elíptico

# Forzar orden
##############
force_order = -1
# force_order = 2


if aprox_name == 'Butterworth':

    # eps, this_order = min_order_butter(omega_p, alfa_max, omega_s, alfa_min)
    # omega_butter = eps**(-1/this_order)

    if force_order > 0:
        this_order = force_order
    else:
        this_order, omega_butter = sig.buttord(omega_p, omega_s, alfa_max, alfa_min, analog=True)
    
    z,p,k = sig.buttap(this_order)
    
    eps = np.sqrt( 10**(alfa_max/10) - 1 )
    
    z,p,k = sig.lp2lp_zpk(z, p, k, omega_butter)


elif aprox_name == 'Chebyshev1':

    if force_order > 0:
        this_order = force_order
    else:
        this_order, _ = sig.cheb1ord(omega_p, omega_s, alfa_max, alfa_min, analog=True)

    z,p,k = sig.cheb1ap(this_order, alfa_max)
    
elif aprox_name == 'Chebyshev2':

    if force_order > 0:
        this_order = force_order
    else:
        this_order, _  = sig.cheb2ord(omega_p, omega_s, alfa_max, alfa_min, analog=True)
    
    z,p,k = sig.cheb2ap(this_order, alfa_min)
    
elif aprox_name == 'Bessel':

    if force_order > 0:
        this_order = force_order
    else:
        this_order = besselord(omega_p, omega_s, alfa_max, alfa_min, omega_d, max_pc_delay)
    
    z,p,k = sig.besselap(this_order, norm='mag')
    
elif aprox_name == 'Cauer':

    if force_order > 0:
        this_order = force_order
    else:
        this_order, _ = sig.ellipord(omega_p, omega_s, alfa_max, alfa_min, analog=True)

    z,p,k = sig.ellipap(this_order, alfa_max, alfa_min)


this_lti = sig.ZerosPolesGain(z, p, k).to_tf()

pretty_print_lti(this_lti)

analyze_sys( [this_lti], [aprox_name ])
