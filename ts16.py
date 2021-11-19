#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS16 ej 2

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor



# Resolución simbólica

s = sp.symbols('s ', complex=True)

z1 = (2*s**3+10.40640532*s**2+21.70820393*s+15)/(1.59359468*s**2+8.29179607*s+15)


# remoción parcial en infinito de 1/z22
z3, l1 = tc2.remover_polo_infinito(z1)

# # remoción total en +/-j3
# z5, dosk1 = tc2.remover_polo_jw(1/y3, 3)

# l1, c2 = tc2.tanque_z(dosk1, sp.Rational('9') )

# # remoción total en infinito
# z7, c1 = tc2.remover_polo_infinito(1/z5)


# print('Verificación')

# Tc1 = sp.Matrix([[1, 0], [s*c1, 1]])
# z4 = dosk1*s/(s**2+9)
# Tc1l1 = sp.Matrix([[1, z4], [0, 1]])
# Tc3yl = sp.Matrix([[1, 0], [s*c3+1, 1]])

# tt =  Tc1 * Tc1l1 * Tc3yl

# zverif = sp.factor(sp.simplify(sp.expand(1/tt[1,0])))

# display( zverif )

# # Dibujo de la red sintetizada

# d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

# d = tc2.dibujar_puerto_entrada(d,
#                                port_name = 'In', 
#                                voltage_lbl = ('+', '$V_1$', '-'), 
#                                current_lbl = '$I_1$')

# d = tc2.dibujar_elemento_derivacion(d, Capacitor, c1)

# d, z5_lbl = tc2.dibujar_funcion_exc_abajo(d, 
#                                           'Z_{5}',  
#                                           z5, 
#                                           hacia_entrada = True,
#                                           k_gap_width = 0.5)

# d = tc2.dibujar_tanque_serie(d, c2, l1)

# d, z3_lbl = tc2.dibujar_funcion_exc_abajo(d, 
#                                           'Z_{3}',  
#                                           1/y3, 
#                                           hacia_entrada = True,
#                                           k_gap_width = 0.5)

# d = tc2.dibujar_elemento_derivacion(d, Capacitor, c3)

# d, z22_lbl = tc2.dibujar_funcion_exc_arriba(d, 
#                                             'Z_{22}',  
#                                             z22, 
#                                             hacia_entrada  = True, 
#                                             k_gap_width = 0.5)

# d = tc2.dibujar_puerto_salida(d, 
#                               port_name = 'Out', 
#                               current_lbl = '$I_2$' )


# display(d)



