#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt

from pytc2.sistemas_lineales import analyze_sys, pretty_print_lti, tf2sos_analog, pretty_print_SOS


num = np.array([1, 0, 2.5, 0, 0.5625, 0])
den = np.array([1, 0.390, 3.067 , 0.785, 3.056, 0.387, 0.989])

# ganancia original para w = 1 en dB
k0db = 41.3

# k0 deseada en el ejemplo en dB
k0_deseada = 10 

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

