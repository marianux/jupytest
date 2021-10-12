#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:38:55 2021

@author: mariano
"""

import sympy as sp


def simplify_n_monic(tt):
    
    num, den = sp.fraction(sp.simplify(tt))
    
    num = sp.poly(num,s)
    den = sp.poly(den,s)
    
    lcnum = sp.LC(num)
    lcden = sp.LC(den)
    
    return( sp.simplify(lcnum/lcden) * (sp.monic(num) / sp.monic(den)) )

# T puenteado
z1, z2, z3 = sp.symbols('z1 z2 z3', complex=True)

Ya = sp.Matrix([[1/z3, -1/z3], [-1/z3, 1/z3]])
Zb = sp.Matrix([[z1+z2, z2], [z2, z1+z2]])

Ztpuenteado = (Ya+Zb**-1)**-1

# doble T
z4, z5 = sp.symbols('z4 z5', complex=True)
R, C, q, m, k = sp.symbols('R C q m k', real=True, positive=True)
s = sp.symbols('s', complex=True)

z1 = 1/(s*q*C)
z2 = R/2

z5 = 1/(s*2*q*C)
z4 = R

Zb = sp.Matrix([[z1+z2, z2], [z2, z1+z2]])
Zc = sp.Matrix([[z4+z5, z5], [z5, z4+z5]])

YdobleT = Zc**-1+Zb**-1

# V2/V1 bicuad notch. Tabla 5-4 fila 2
Tbn = simplify_n_monic( YdobleT[1,0] / (YdobleT[1,1] *(k-1)/k + YdobleT[1,0] - s*m*C/k ))

display(Tbn)
