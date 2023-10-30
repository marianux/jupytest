#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:44:42 2023

@author: mariano
"""


import sympy as sp

# Ahora importamos las funciones de PyTC2

from pytc2.remociones import isFRP, remover_valor_en_infinito, remover_polo_infinito
from pytc2.sintesis_dipolo import cauer_RC, foster, foster_zRC2yRC
from pytc2.dibujar import display, dibujar_cierre, dibujar_puerto_entrada, dibujar_funcion_exc_abajo,  dibujar_elemento_serie, dibujar_elemento_derivacion,  dibujar_tanque_derivacion, dibujar_tanque_RC_serie,  dibujar_espacio_derivacion, Capacitor, Resistor, Inductor
from pytc2.general import print_latex, print_subtitle, expr_simb_expr, a_equal_b_latex_s
from IPython.display import display, Markdown

# Resolución simbólica
s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2 + 2*s + 2)/(s**2 + 1*s + 1)

print_subtitle('Impedancia $Z_{RLC}$ ')

print_latex(a_equal_b_latex_s('Z_{RLC}(s)', ZZ))

Z2, R1 = remover_valor_en_infinito(ZZ)

Y4, Y3 = remover_polo_infinito(1/Z2)

C1 = Y3/s

Z6, Z5 = remover_polo_infinito(1/Y4)

L1 = Z5/s

R2 = Z6

d = dibujar_puerto_entrada('',
                        voltage_lbl = ('+', '$V1$', '-'), 
                        current_lbl = '$I1$')

d, zz_lbl = dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          ZZ, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = dibujar_elemento_serie(d, Resistor, R1)

d = dibujar_elemento_derivacion(d, Capacitor, C1)

d = dibujar_elemento_serie(d, Inductor, L1)

d = dibujar_elemento_serie(d, Resistor, R2)

d = dibujar_cierre(d)

display(d)



