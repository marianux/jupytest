#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS11 ej 1

@author: mariano
"""

import sympy as sp
from pytc2.remociones import remover_valor, remover_polo_sigma
from pytc2.dibujar import display, dibujar_puerto_entrada, dibujar_funcion_exc_abajo,  dibujar_elemento_serie, dibujar_espacio_derivacion, Resistor,dibujar_tanque_RC_derivacion, dibujar_elemento_derivacion


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
    
Z2, Ra = remover_valor(ZZ, sigma_zero = sigma1)

Y4, Y3, R1, C1 = remover_polo_sigma(1/Z2, sigma1, isImpedance = False)

Z6, Rb = remover_valor(1/Y4, sigma_zero = sigma2)

Y8, k2, R2, C2 = remover_polo_sigma(1/Z6, sigma2, isImpedance = False)

Rc = 1/Y8

# Dibujamos la red resultante:

d = dibujar_puerto_entrada('',
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          ZZ, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = dibujar_elemento_serie(d, Resistor, Ra)

d = dibujar_tanque_RC_derivacion(d, R1, C1)
    
d = dibujar_elemento_serie(d, Resistor, Rb)

d = dibujar_tanque_RC_derivacion(d, R2, C2)
                        
d = dibujar_espacio_derivacion(d)

d = dibujar_elemento_derivacion(d, Resistor, Rc)

display(d)

