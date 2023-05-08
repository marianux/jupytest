#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano
"""

import sympy as sp
from sympy.abc import s
from IPython.display import display, Math

from pytc2.sintesis_dipolo import foster
from pytc2.remociones import remover_polo_dc, remover_polo_infinito

I1, V1, V2, V3, V4, V5 = sp.symbols("I1, V1, V2, V3, V4, V5")
Y1, Y2, Y3, Y4, Y5, As, wt, w0 = sp.symbols("Y1, Y2, Y3, Y4, Y5, As, wt, w0")
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

print('#############################################################')
print('# Z1 para el GIC de Antoniou sin considerar realim negativa #')
print('#############################################################')

display(Math( r' Z_1^{ir} = ' + sp.latex(Z1_opamp_ideal) ))


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

# Si G/C = w0 = 1; wt = 1000 y G = 1
z1_real = (s**3 + s**2 * 1001 + s * (10**6/2+10**3) ) / (s**3 + s**2 * 1001 + s * 10**3 + 10**6/2 )

Ginf = sp.limit(1/z1_real, s, sp.oo)
G0 = sp.limit(1/z1_real, s, 0)

R1 = 1/Ginf

# remuevo la menor admitancia
Y2 = sp.factor(sp.simplify(sp.expand(1/z1_real - Ginf)))

Y4, Yl1 = remover_polo_dc(Y2)

# Zl1  es la impedancia removida
# extraigo L1
L1 = Yl1/s


