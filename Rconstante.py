#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:11:13 2022

@author: mariano

Red de resistencia constante implementada con T puenteado
"""

import sympy as sp
import splane as tc2


# T puenteado
Zb, Za = sp.symbols('Zb Za', complex=True)
R = sp.symbols('R', real=True, positive=True)

Ztee = sp.Matrix([[R+Zb, Zb], [Zb, Zb+R]])
Ypuente = sp.Matrix([[1/Za, -1/Za], [-1/Za, 1/Za]])

Ztpuenteado = (Ypuente+Ztee**-1)**-1


Tgen = sp.Matrix([[1, R], [0, 1]])
Tcarga = sp.Matrix([[1, 0], [1/R, 1]])

Ttp = tc2.Z2T_s(Ztpuenteado)

Trcons =   Ttp * Tcarga
Trcons = Trcons.subs(Za*Zb, R**2)
# Trcons = Trcons.subs(Za, R**2/Zb)
Trcons = Trcons.subs(Zb, R**2/Za)

Vtransf = 1/(tc2.simplify_n_monic(Trcons[0,0]))
