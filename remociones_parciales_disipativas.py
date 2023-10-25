#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 2

@author: mariano
"""

import sympy as sp
from pytc2.remociones import remover_polo_dc, remover_polo_jw
from pytc2.remociones import remover_valor_en_dc, remover_valor_en_infinito, remover_polo_sigma
from pytc2.dibujar import display, dibujar_puerto_entrada, dibujar_funcion_exc_abajo,  dibujar_elemento_serie, dibujar_elemento_derivacion,  dibujar_tanque_derivacion, dibujar_tanque_RC_serie,  dibujar_espacio_derivacion, Capacitor, ResistorIEC
from pytc2.general import print_latex, print_subtitle, a_equal_b_latex_s


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2 + 13*s + 32)/(3*s**2 + 27*s+ 44)

# Red ejemplo 2
d = dibujar_puerto_entrada('',
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = dibujar_funcion_exc_abajo(d, 
                 'Z(s)',  
                 ZZ, 
                 hacia_salida = True,
                 k_gap_width = 0.5)

d = dibujar_elemento_derivacion(d, ResistorIEC, 'Y_A')

d = dibujar_espacio_derivacion(d)

d = dibujar_tanque_RC_serie(d, sym_R_label='R_1', capacitor_lbl='C_1' )

d = dibujar_espacio_derivacion(d)

d = dibujar_elemento_derivacion(d, ResistorIEC, 'Y_B')

d = dibujar_espacio_derivacion(d)

d = dibujar_tanque_RC_serie(d, sym_R_label='R_2', capacitor_lbl='C_2' )

d = dibujar_espacio_derivacion(d)

d = dibujar_elemento_derivacion(d, ResistorIEC, 'Y_C')


display(d)


sigma_R1C1 = -1

Y2, YA = remover_valor_en_dc(1/ZZ, sigma_zero = sigma_R1C1 )

print_latex(a_equal_b_latex_s('Y_A', YA))
print_latex(a_equal_b_latex_s('Y_2', Y2))

# removemos R1-C1
Z4, ZR1C1, R1, C1 = remover_polo_sigma(1/Y2, sigma = sigma_R1C1, isImpedance = True, isRC = True )

print_latex(a_equal_b_latex_s('Z_3', ZR1C1))
print_latex(a_equal_b_latex_s('Z_4', Z4))



