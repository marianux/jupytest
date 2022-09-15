#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 2

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor
import numpy as np



# Resolución simbólica

s = sp.symbols('s ', complex=True)

YY = 3*s*(s**2+sp.Rational(7,3))/(s**2+2)/(s**2+5)

# Restricción circuital: L2*C2 = 1 r/s
# remoción parcial en infinito de 1/YY

omega_L2C2 = 1

Z2, Zc1 = tc2.remover_polo_dc(1/YY, omega_zero = omega_L2C2 )

# Yc1 es la admitancia removida
# extraigo C1
C1 = 1/(s*Zc1)

Y4, Zt2, L2, C2 = tc2.remover_polo_jw(1/Z2, isImpedance = False, omega = omega_L2C2 )

# quedaría solo un tanque en Y4, no especifico omega.
Y6, Zt3, L3, C3 = tc2.remover_polo_jw(Y4, isImpedance = False)


# Dibujamos la red resultante:
    
d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d = tc2.dibujar_puerto_entrada(d,
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Y',  
                                          YY, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_elemento_serie(d, Capacitor, C1)

d = tc2.dibujar_tanque_derivacion(d, L2, C2)

d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_tanque_derivacion(d, L3, C3)

display(d)
