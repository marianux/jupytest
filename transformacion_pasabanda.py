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
from splane import simplify_n_monic
from IPython.display import display, Math

w_o, Q, Q_bp, B_bp, W_o_bp = sp.symbols("w_o, Q, Q_bp, B_bp, W_o_bp")


# print('##############################')
# print('# Z1 para el GIC de Antoniou #')
# print('##############################')



############################################
# Definicion de las funciones transferencia. 

H1 = 1/(s+w_o)

H2 = w_o**2/(s**2 + s*w_o/Q + w_o**2)

# nucleo de transformación pasabanda
Kbp = Q_bp * (s**2 + W_o_bp**2) / (W_o_bp*s)

H1bp = simplify_n_monic(H1.subs(s, Kbp))

H2bp = simplify_n_monic(H2.subs(s, Kbp))



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
