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
from splane import simplify_n_monic, parametrize_sos
from IPython.display import display, Math

w_o, Q, Q_bp, B_bp, W_o_bp = sp.symbols("w_o, Q, Q_bp, B_bp, W_o_bp")
a, b, c, d, e, f = sp.symbols("a, b, c, d, e, f")


# print('##############################')
# print('# Z1 para el GIC de Antoniou #')
# print('##############################')



############################################
# Definicion de las funciones transferencia. 

H1 = 1/(s+w_o)

# H2 = w_o**2/(s**2 + s*w_o/Q + w_o**2)
H2 = b/(s**2 + s*a + b)

# nucleo de transformación pasabanda
Kbp = Q_bp * (s**2 + 1) / s

H1bp = sp.simplify(sp.expand(H1.subs(s, Kbp)))
num, den = sp.fraction(H1bp)
num = sp.Poly(num,s)
den = sp.Poly(den,s)
H1bp, w_on, Q_n, w_od, Q_d, K  = parametrize_sos(num, den)

H2bp = sp.simplify(sp.expand(H2.subs(s, Kbp)))
num, den = sp.fraction(H2bp)
num = sp.Poly(num,s)
den = sp.Poly(den,s)
H2bp, w_on, Q_n, w_od, Q_d, K  = parametrize_sos(num, den)



# print('#############################################################')
# print('# Z1 para el GIC de Antoniou sin considerar realim negativa #')
# print('#############################################################')

# display(Math( r' Z_1^{ir} = ' + sp.latex(Z1_opamp_ideal) ))


# # modelo integrador A(s)=\omega_t/s (sin asumir realimentación negativa)
# Z1 = sp.simplify(sp.expand(Z1.subs(As, wt/s)))
# Z1 = sp.simplify(sp.expand(Z1.subs({Y1:G, Y2:s*C, Y3:G, Y4:G, Y5:G})))
                      
# num, den = sp.fraction(Z1)

# num = sp.Poly(num,s)
# den = sp.Poly(den,s)

# k = num.LC() / den.LC()

# num = num.monic()
# den = den.monic()

# # Implementación de un inductor mediante GIC con modelo real
# Z1_opamp_real  = num/den*k

# # ¿Qué tipo de Z1 será?

# print('#################################################')
# print('# Z1 para el GIC de Antoniou (OpAmp integrador) #')
# print('#################################################')
# display(Math( r' Z_1^r = ' + sp.latex(Z1_opamp_real) ))

# print('¿Qué tipo de Z1 será?')
