#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:11:13 2022

@author: mariano

Red de resistencia constante implementada con lattice
"""

import sympy as sp

from pytc2.cuadripolos import Z2Tabcd_s, Tabcd2Z_s
from pytc2.general import print_latex, a_equal_b_latex_s, simplify_n_monic


Zb, Za = sp.symbols('Zb Za', complex=True)
R = sp.symbols('R', real=True, positive=True)

Zlatt= sp.Matrix([
                     [(Za+Zb)/2, (Zb-Za)/2], 
                     [(Zb-Za)/2, (Za+Zb)/2]]
    )

Tgen = sp.Matrix([[1, R], [0, 1]])
Tcarga = sp.Matrix([[1, 0], [1/R, 1]])

Tlatt = Z2Tabcd_s(Zlatt)

# Ojo que la transferencia es Vo/Vg.
Trcons =   Tgen * Tlatt * Tcarga
Trcons = Trcons.subs(Za*Zb, R**2)
# Trcons = Trcons.subs(Za, R**2/Zb)
Trcons = Trcons.subs(Zb, R**2/Za)

k, num, den = simplify_n_monic(Trcons[0,0]) 

Atot = k * num / den

Vtransf = 1/Atot

print_latex(a_equal_b_latex_s('\\frac{V_2}{V_g}', Vtransf))

# Ojo que quiero ver la impedancia a la entrada
# del cuadripolo cargado
p1 =   Tlatt * Tcarga
p1 = p1.subs(Za*Zb, R**2)
# p1 = p1.subs(Zb, (R**2)/Za)

z1 = sp.simplify(sp.expand(p1[0,0]/p1[1,0]))

print_latex(a_equal_b_latex_s('Z_1', z1 ))

