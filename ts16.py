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

Ts = sp.Rational('15')/(s**3 + sp.Rational('6')*s**2 + sp.Rational('15')*s + sp.Rational('15') )

s11sq = sp.factor( 1- (Ts * Ts.subs(s, -s)) )

s11 = tc2.modsq2mod_s(s11sq)

z1 = sp.simplify(sp.expand(sp.simplify(sp.expand(1+s11))/sp.simplify(sp.expand((1-s11)))))

# remoción parcial en infinito de z1
z3, l1 = tc2.remover_polo_infinito(z1)

# remoción parcial en infinito de y3
y5, c1 = tc2.remover_polo_infinito(1/z3)

# remoción parcial en infinito de z5
z7, l2 = tc2.remover_polo_infinito(1/y5)


# Dibujo de la red sintetizada

d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d, z1_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z_{1}',  
                                          z1.evalf(4), 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_puerto_entrada(d)

d = tc2.dibujar_elemento_serie(d, Inductor, l1.evalf(4))

d = tc2.dibujar_elemento_derivacion(d, Capacitor, c1.evalf(4))

d = tc2.dibujar_elemento_serie(d, Inductor, l2.evalf(4))

d = tc2.dibujar_puerto_salida(d)

d = tc2.dibujar_espaciador(d)

d = tc2.dibujar_elemento_derivacion(d, Resistor, z7)

display(d)



