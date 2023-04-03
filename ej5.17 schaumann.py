#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


from pytc2.sistemas_lineales import analyze_sys, pretty_print_lti, tf2sos_analog, pretty_print_SOS


num = np.polymul([1, 0, 29.2**2], [1, 0, 43.2**2])
den = np.polymul(np.polymul([1, 16.8], [1, 19.4, 20.01**2]), [1, 4.72, 22.52**2])

pretty_print_lti(num, den)

# find maximum in transfer function
thisFilter = sig.TransferFunction(num, den)

# analyze_sys( thisFilter, 'Ej 5.17' )

_, mag, _ = thisFilter.bode(np.logspace(-1,1,1000))

# bode in dB
k0db = np.max(mag)

# k0 deseada en el ejemplo en dB
k0_deseada = 13.5

# factor de corrección de ganancia
k_corr = 10**((k0_deseada - k0db)/20)

num = k_corr * num

print('\n\n')
print('----------------------')
print('Transferencia original')
print('----------------------')
pretty_print_lti(num, den)

# particiono en SOS's para la implementación
sos_pb = tf2sos_analog(num, den)

# ajusto posibles problemas numericos (negativos y valores despreciables)
# asumimos siempre transferencias normalizadas.
sos_pb[sos_pb < 1e-6] = 0.0

# parametrizada
pretty_print_SOS(sos_pb)

# parametrizada
pretty_print_SOS(sos_pb, mode='omegayq')

plt.close('all')

# analyze_sys( sig.TransferFunction(num, den) )
analyze_sys( sos_pb )

