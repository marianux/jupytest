#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano
"""

import sympy as sp
from scipy.signal import TransferFunction


from pytc2.sintesis_dipolo import foster
from pytc2.remociones import remover_polo_dc, remover_polo_infinito, isFRP
from pytc2.general import a_equal_b_latex_s, print_latex, s
from pytc2.sistemas_lineales import analyze_sys

I1, V1, V2, V3, V4, V5 = sp.symbols("I1, V1, V2, V3, V4, V5")
Y1, Y2, Y3, Y4, Y5, As = sp.symbols("Y1, Y2, Y3, Y4, Y5, As")
G, C, wt, w0 = sp.symbols("G, C, wt, w0", real = True) 

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

# display(Math( r' Z_1^i = ' + sp.latex(Z1) ))
print_latex(a_equal_b_latex_s('Z_1^i', Z1))

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

# display(Math( r' Z_1^{ir} = ' + sp.latex(Z1_opamp_ideal) ))
print_latex(a_equal_b_latex_s('Z_1^{ir}', Z1_opamp_ideal))


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
# display(Math( r' Z_1^r = ' + sp.latex(Z1_opamp_real) ))
print_latex(a_equal_b_latex_s('Z_1^r', Z1_opamp_real))

print('¿Qué tipo de Z1 será?')

ww = sp.symbols("\omega", real = True)

# Si G/C = w0 = 1; wt = 1000 y G = 1
z1_realopamp = sp.simplify(sp.expand(Z1_opamp_real.subs({wt:100, G:1, C:1})))

num, den = sp.fraction(z1_realopamp)

num.as_poly(s).all_coeffs()
    
aa = np.array( [ float(ii) for ii in num.as_poly(s).all_coeffs()]
bb = np.array( [ float(ii) for ii in den.as_poly(s).all_coeffs()]

cc = TransferFunction(aa, bb)
                      
analyze_sys(  )


# re_z1_realopamp = sp.simplify(sp.expand(sp.re(z1_realopamp.subs({s:(sp.I*ww)}))))
# re_y1_realopamp = sp.simplify(sp.expand(sp.re(1/z1_realopamp.subs({s:(sp.I*ww)}))))
re_z1_realopamp = sp.simplify(sp.expand(sp.re(Z1_opamp_real.subs({s:(sp.I*ww)}))))


Goo = sp.limit(1/Z1_opamp_real, s, sp.oo)
G0 = sp.limit(1/Z1_opamp_real, s, 0)

Y2 = sp.factor(sp.simplify(sp.expand(1/Z1_opamp_real - Goo)))


Goo = sp.limit(1/z1_realopamp, s, sp.oo)
G0 = sp.limit(1/z1_realopamp, s, 0)


# remuevo la menor admitancia
Y2 = sp.factor(sp.simplify(sp.expand(1/z1_realopamp - Goo)))

# Y4, Yl1 = remover_polo_dc(Y2)

# Zl1  es la impedancia removida
# extraigo L1
# L1 = Yl1/s


