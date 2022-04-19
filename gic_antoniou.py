#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano
"""

import sympy as sp
from sympy.abc import s
from IPython.display import display, Math

I1, V1, V2, V3, V4, V5 = sp.symbols("I1, V1, V2, V3, V4, V5")
Y1, Y2, Y3, Y4, Y5, As, wt = sp.symbols("Y1, Y2, Y3, Y4, Y5, As, wt")
G, C = sp.symbols("G, C") 

# modelo ideal negativamente realimentado
aa = sp.solve([ 
                V1*Y1 - V2*Y1 - I1, 
                -V2*Y2 + V1*(Y2+Y3) -V3*Y3,
                -V3*Y4 + V1*(Y4+Y5)
                ], 
                [V1, V2, V3])
Z1 = aa[V1]/I1

print('##############################')
print('# Z1 para el GIC de Antoniou #')
print('##############################')

display(Math( r' Z_1^i = ' + sp.latex(Z1) ))


######################################################
# solo modelo ideal, no sabemos qué realimentación es. 
aa = sp.solve([ 
                V1*Y1 - V2*Y1 - I1, 
                -V2*Y2 + V3*(Y2+Y3) -V4*Y3,
                -V4*Y4 + V5*(Y4+Y5),
                As*(V5-V3) - V2, 
                As*(V1-V3) - V4, 
                ], 
                [V1, V2, V3, V4, V5])

Z1 = aa[V1]/I1

# modelo ideal sin asumir realimentación negativa
Z1_opamp_ideal = sp.limit(Z1, As, sp.oo)

# modelo integrador A(s)=\omega_t/s (sin asumir realimentación negativa)
Z1 = sp.simplify(sp.expand(Z1.subs(As, wt/s)))
Z1 = sp.simplify(sp.expand(Z1.subs({Y1:G, Y2:s*C, Y3:G, Y4:G, Y5:G})))
                      
num, den = sp.fraction(Z1)

num = sp.Poly(num,s)
den = sp.Poly(den,s)

k = num.LC() / den.LC()

num = num.monic()
den = den.monic()

# Implementación de un inductor mediante GIC con modelo real
Z1_opamp_real  = num/den*k

# ¿Qué tipo de Z1 será?

print('#################################################')
print('# Z1 para el GIC de Antoniou (OpAmp integrador) #')
print('#################################################')
display(Math( r' Z_1^r = ' + sp.latex(Z1_opamp_real) ))

print('¿Qué tipo de Z1 será?')
