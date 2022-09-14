#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 1

@author: mariano
"""

import sympy as sp
from schemdraw import Drawing
import splane as tc2


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2+6*s+8)/(s**2+4*s+3)

# Halle los valores de los componentes de la topología resultante.

# remoción parcial para que el siguiente tanque R1-C1 resuenen a 6 r/s

# Consignas del ejercicio: resonancias de dos tanques RC
sigma1 = 6
sigma2 = sp.Rational('7/2')

# La topología circuital guía las remociones:
    
Z2, Ra = tc2.remover_valor(ZZ, sigma1)

Y4, k1 = tc2.remover_polo_sigma(sigma1, yy = 1/Z2)

R1 = 1/k1
C1 = k1/sigma1

Z6, Rb = tc2.remover_valor(1/Y4, sigma2)

Y8, k2 = tc2.remover_polo_sigma(sigma2, yy = 1/Z6)

R2 = 1/k2
C2 = k2/sigma2

Rc = 1/Y8

# Dibujamos la red resultante:
    
d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d = tc2.dibujar_puerto_entrada(d,
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          ZZ, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_elemento_serie(d, Resistor, Ra)

d = tc2.dibujar_tanque_RC_derivacion(d, R1, C1)
    
d = tc2.dibujar_elemento_serie(d, Resistor, Rb)

d = tc2.dibujar_tanque_RC_derivacion(d, R2, C2)
                        
d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_elemento_derivacion(d, Resistor, Rb)

display(d)

