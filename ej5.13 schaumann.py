#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:44:07 2023

@author: mariano
"""

import numpy as np
import scipy.signal as sig

# Ahora importamos las funciones de PyTC2

from pytc2.sistemas_lineales import analyze_sys, pretty_print_lti, tf2sos_analog, pretty_print_SOS
from pytc2.general import print_subtitle
from IPython.display import display, Markdown

# Comienzo de la simulación

# definimos la función transferencia
num = np.array([1, 0, 2.5, 0, 0.5625, 0])
den = np.array([1, 0.390, 3.067 , 0.785, 3.056, 0.387, 0.989])

pretty_print_lti(num, den)

# find maximum in transfer function
thisFilter = sig.TransferFunction(num, den)

# analyze_sys( thisFilter, 'Ej 5.13' )

_, mag, _ = thisFilter.bode(np.logspace(-1,1,1000))

# bode in dB
k0db = np.max(mag)

# ganancia original para w = 1 en dB (por observación)
# k0db = 41.3

# k0 deseada en el ejemplo en dB
k0_deseada = 10 

# factor de corrección de ganancia
k_corr = 10**((k0_deseada - k0db)/20)

# corrección de ganancia
num = k_corr * num



display(Markdown('### Ejemplo 5.13 del Schaumann: Factorización en secciones de segundo orden' ))

print_subtitle('Función transferencia con 10 dB')

pretty_print_lti(num, den)

# particiono en SOS's para la implementación
sos_pb = tf2sos_analog(num, den)

# ajusto posibles problemas numericos (negativos y valores despreciables)
# asumimos siempre transferencias normalizadas.
sos_pb[sos_pb < 1e-6] = 0.0

print_subtitle('Transferencia factorizada')

# visualizamos la transferencia factorizada
pretty_print_SOS(sos_pb)

print_subtitle('Transferencia factorizada y parametrizada')

# visualizamos la transferencia factorizada y parametrizada
pretty_print_SOS(sos_pb, mode='omegayq')

analyze_sys( sos_pb, 'Ej 5.13' )
