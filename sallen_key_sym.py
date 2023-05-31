#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano
"""

import sympy as sp
from IPython.display import display, Math

# Ahora importamos las funciones de PyTC2
from pytc2.sistemas_lineales import simplify_n_monic, parametrize_sos


# variables simbólicas
s = sp.symbols('s', complex=True)
Va, Vb, Vi, Vo = sp.symbols("Va, Vb, Vi, Vo")
G1, G2, Ga, Gb, C, As, wt = sp.symbols("G1, G2, Ga, Gb, C, As, wt")

# Sistemas de ecuaciones del modelo ideal, negativamente realimentado
aa = sp.solve([ 
                Vb*(G1 + G2 + s*C) - Vi * G1 - Va * G2 - Vo * s*C, 
                -Vb*G2 + Va * (G2 + s*C),
                -Vo*Gb + Va * (Ga + Gb)
                ], 
                [Vo, Vi, Va, Vb])
T1 = aa[Vo]/aa[Vi]

num, den = sp.fraction(sp.simplify(sp.expand(T1)))
num = sp.Poly(num,s)
den = sp.Poly(den,s)
num1, den1, w_on, Q_n, w_od, Q_d, k1 = parametrize_sos(num, den)

display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(sp.Mul(k1,num1/den1, evaluate=False)) ))


# # Ejercicio 7.a: Pasatodo de 1er orden

# tf7a = transf_func.subs(Y1, s*C)
# tf7a = tf7a.subs(Y2, 1/R)

# num, den = sp.fraction(sp.simplify(sp.expand(tf7a)))

# num = sp.Poly(num,s)
# den = sp.Poly(den,s)

# k = num.LC() / den.LC()

# num = num.monic()
# den = den.monic()

# den_coeffs = den.all_coeffs()
# wo = den_coeffs[-1]

# tf7a_final = sp.Mul(k,num/den, evaluate=False)

# print('')
# print('################')
# print('# Ejercicio 7a #')
# print('################')
# display(tf7a_final)
# display(Math( r' \omega_o = ' + sp.latex(wo) ))

# # Ejercicio 7.b: Pasatodo de 2do orden

# tf7b = transf_func.subs(Y1, 1/(R+1/s/C))
# tf7b = tf7b.subs(Y2, 1/R+s*C)

# num, den = sp.fraction(sp.simplify(sp.expand(tf7b)))

# num = sp.Poly(num,s)
# den = sp.Poly(den,s)

# k = num.LC() / den.LC()

# num = num.monic()
# den = den.monic()

# ## parametrizamos en función de \omega_o y Q
# den_coeffs = den.all_coeffs()
# wo_sq = den_coeffs[-1]
# qq = sp.powsimp(sp.sqrt(1/(den_coeffs[1]**2 / wo_sq)))

# tf7b_final = sp.Mul(k,num/den, evaluate=False)

# print('')
# print('')
# print('################')
# print('# Ejercicio 7b #')
# print('################')
# display(tf7b_final)
# display(Math( r' \omega_o^2 = ' + sp.latex(wo_sq) ))
# display(Math( r' Q = ' + sp.latex(qq) ))

