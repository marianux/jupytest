#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:11:13 2022

@author: mariano

Red de resistencia constante implementada con lattice
"""

import sympy as sp

from pytc2.cuadripolos import Z2Tabcd_s
from pytc2.sistemas_lineales import simplify_n_monic
from pytc2.general import print_latex, a_equal_b_latex_s


# T puenteado
Zb, Za = sp.symbols('Zb Za', complex=True)
R = sp.symbols('R', real=True, positive=True)

Zlatt= sp.Matrix([[(Za+Zb)/2, (Zb-Za)/2], [(Zb-Za)/2, (Za+Zb)/2]])

Tgen = sp.Matrix([[1, R], [0, 1]])
Tcarga = sp.Matrix([[1, 0], [1/R, 1]])

Tlatt = Z2Tabcd_s(Zlatt)

Trcons =   Tgen * Tlatt * Tcarga
Trcons = Trcons.subs(Za*Zb, R**2)
# Trcons = Trcons.subs(Za, R**2/Zb)
Trcons = Trcons.subs(Zb, R**2/Za)

Vtransf = 1/(simplify_n_monic(Trcons[0,0]))

print_latex(a_equal_b_latex_s('\\frac{V_2}{V_1}', Vtransf))

p1 =   Tlatt * Tcarga
p1 = p1.subs(Za*Zb, R**2)
# p1 = p1.subs(Zb, (R**2)/Za)

# z1 = sp.simplify(sp.expand((p1[0,0]*R + p1[0,1])/(p1[1,0]*R + p1[1,1])))

# z1 = sp.simplify(sp.expand(Zlatt[0,0] - Zlatt[1,0]**2/(Zlatt[1,1]+R)))
# z1 = z1.subs(Za*Zb, R**2)
# z1 = sp.simplify(sp.expand(z1.subs(Zb, (R**2)/Za)))


print_latex(a_equal_b_latex_s('Z_1',  ))

