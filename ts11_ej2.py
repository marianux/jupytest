#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS11 ej 2

@author: mariano
"""

import sympy as sp
import numpy as np

from pytc2.dibujar import display, dibujar_puerto_entrada, dibujar_funcion_exc_abajo, dibujar_elemento_serie, dibujar_espacio_derivacion, Resistor, Capacitor, Inductor, dibujar_elemento_derivacion
from pytc2.remociones import remover_polo_infinito

# Resolución simbólica

s = sp.symbols('s ', complex=True)

ZZ = (s**2+s+1)/(s**2+2*s+5)/(s+1)

# remoción total en infinito de 1/ZZ

Y2, Yc1 = remover_polo_infinito(1/ZZ)

# Yc1 es la admitancia removida
# extraigo C1
C1 = Yc1/s

Ginf = sp.limit(Y2, s, sp.oo)
G0 = sp.limit(Y2, s, 0)

# remuevo la menor admitancia
R1 = 1/np.min((Ginf, G0))
Y4 = sp.factor(sp.simplify(sp.expand(Y2 - 1/R1)))

Z6, Zl1 = remover_polo_infinito(1/Y4)

# Zl1  es la impedancia removida
# extraigo L1
L1 = Zl1/s

# remuevo la menor resistencia
Rinf = sp.limit(Z6, s, sp.oo)
R0 = sp.limit(Z6, s, 0)
R2 = np.min((Rinf, R0))
Z8 = sp.factor(sp.simplify(sp.expand(Z6 - R2)))

# extraigo C2
C2 = sp.limit(1/s/Z8, s, sp.oo)

Y10, Yc2 = remover_polo_infinito(1/Z8)
# Yc1 es la admitancia removida
# extraigo C1
C2 = Yc2/s

R3 = 1/Y10



# Dibujamos la red resultante:

d = dibujar_puerto_entrada('',
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          ZZ, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = dibujar_elemento_derivacion(d, Capacitor, C1)

d = dibujar_espacio_derivacion(d)

d = dibujar_elemento_derivacion(d, Resistor, R1)
    
d = dibujar_elemento_serie(d, Inductor, L1)

d = dibujar_elemento_serie(d, Resistor, R2)

d = dibujar_elemento_derivacion(d, Capacitor, C2)

d = dibujar_espacio_derivacion(d)

d = dibujar_elemento_derivacion(d, Resistor, R3)

display(d)
