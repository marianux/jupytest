#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS13 ej 2

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor



# Resolución simbólica

s = sp.symbols('s ', complex=True)

z22 = (2*s**2+1)/(s**3+2*s)


# remoción parcial en infinito de 1/z22
y3, Yc3 = tc2.remover_polo_infinito(1/z22, omega_zero=3)

c3 = Yc3/s

# remoción total en +/-j3
z5, z4, l1, c2 = tc2.remover_polo_jw(1/y3, omega = 3, isImpedance = True  )

# remoción total en infinito
y7, Yc1 = tc2.remover_polo_infinito(1/z5)

c1 = Yc1/s

print('Verificación')

Tc1 = sp.Matrix([[1, 0], [Yc1, 1]])
Tc1l1 = sp.Matrix([[1, z4], [0, 1]])
Tc3yl = sp.Matrix([[1, 0], [Yc3+1, 1]])

tt =  Tc1 * Tc1l1 * Tc3yl

zverif = sp.factor(sp.simplify(sp.expand(1/tt[1,0])))

display( zverif )

# Dibujo de la red sintetizada

d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d = tc2.dibujar_puerto_entrada(d,
                               port_name = 'In', 
                               voltage_lbl = ('+', '$V_1$', '-'), 
                               current_lbl = '$I_1$')

d = tc2.dibujar_elemento_derivacion(d, Capacitor, c1)

d, z5_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z_{5}',  
                                          z5, 
                                          hacia_entrada = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_tanque_serie(d, c2, l1)

d, z3_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z_{3}',  
                                          1/y3, 
                                          hacia_entrada = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_elemento_derivacion(d, Capacitor, c3)

d, z22_lbl = tc2.dibujar_funcion_exc_arriba(d, 
                                            'Z_{22}',  
                                            z22, 
                                            hacia_entrada  = True, 
                                            k_gap_width = 0.5)

d = tc2.dibujar_puerto_salida(d, 
                              port_name = 'Out', 
                              current_lbl = '$I_2$' )


display(d)



