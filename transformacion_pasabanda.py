#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano

Script para analizar simbólicamente la transformación pasabanda
aplicada a un sistema de primer y segundo orden.
"""

import sympy as sp
from sympy.abc import s
from pytc2.sistemas_lineales import parametrize_sos
from pytc2.general import print_latex


ww, qq, Q_bp, B_bp, W_o_bp, w_o1, w_o2, qq1, qq2 = sp.symbols("w_o, q_o, Q_bp, B_bp, W_o_bp, w_1, w_2, q_1, q_2")
a, b, c, d, e, f = sp.symbols("a, b, c, d, e, f")
p1 = sp.symbols("p1", complex=True )


# print('##############################')
# print('# Z1 para el GIC de Antoniou #')
# print('##############################')



############################################
# Definicion de las funciones transferencia. 

H1 = 1/(s+ww)

# H2 = w_o**2/(s + p1 )/(s + sp.conjugate(p1))
# H2 = ww**2/(s**2 + s*ww/qq + ww**2)
H2 = b/(s**2 + s*a + b)


# nucleo de transformación pasabanda
Kbp = Q_bp * (s**2 + 1) / s


# Análisis de la transformación pasabanda para una transferencia de primer orden.
H1bp = sp.simplify(sp.expand(H1.subs(s, Kbp)))
num, den = sp.fraction(H1bp)
num = sp.Poly(num,s)
den = sp.Poly(den,s)
num1_bp, den1_bp, w1_on, Q1_n, w1_od, Q1_d, K1_bp  = parametrize_sos(num, den)

print('\n\n')
print('----------------')
print('* Primer orden *')
print('----------------')

print_latex('$ T(s) = ' + sp.latex(H1) + '$')
print_latex('$ T(s) |_{s = ' + sp.latex(Kbp) +  ' } = ' + sp.latex(K1_bp) + ' \\frac{ ' + sp.latex(num1_bp.as_expr()) + '}{' + sp.latex(den1_bp.as_expr()) + '} $')

# Análisis de la transformación pasabanda para una SOS.
H2bp = sp.simplify(sp.expand(H2.subs(s, Kbp)))
num2_bp, den2_bp = sp.fraction(H2bp)

num2_bp = sp.Poly(num2_bp, s)
den2_bp = sp.Poly(den2_bp,s)

# planteo una parametrización en dos polinomios de 2do orden
# de igual Q y wo recíproco respecto a la wo = 1 del kernel 
# de la transformación pasabanda, para ayudar al solver a
# encontrar una solución cerrada a qq1
den1 = s**2 + s*w_o1/qq1 + w_o1**2
den2 = s**2 + s*w_o2/qq2 + w_o2**2

# den2 = den2.subs(w_o2, 1/w_o1)
# den2 = den2.subs(qq2, qq1)

den4 = den1 * den2

den4 = sp.Poly(den4,s)

den_bp_lc = den2_bp.LC()
den2_bp = den2_bp.monic()

den_bp_coeffs = den2_bp.all_coeffs()
den4_coeffs = den4.all_coeffs()

ff = [ aa - bb for aa, bb in zip(den4_coeffs, den_bp_coeffs)]

aa = sp.solve(ff[1:], [ qq1, w_o1 , qq2], dict=True)

print('\n\n')
print('-----------------')
print('* Segundo orden *')
print('-----------------')

print_latex('$ T(s) = ' + sp.latex(H2) + '$')

print_latex('$ T(s) |_{s = ' + sp.latex(Kbp) +  ' } = ' + sp.latex(sp.expand(H2bp)) + '$')
print_latex('$ T(s) |_{s = ' + sp.latex(Kbp) +  ' } =  \\frac{ \\frac{s w_o}{Q_{bp}} }{' + sp.latex(den1) + '}     \\frac{ \\frac{s w_o}{Q_{bp}} }{' + sp.latex(den2) + '} $')

print_latex('$ q_1 = ' + sp.latex(aa[0][qq]) + '$')


