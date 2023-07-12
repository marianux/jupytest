#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:06:52 2023

@author: mariano
"""

from pytc2.dibujar import dibujar_Pi, dibujar_Tee, dibujar_elemento_serie, dibujar_elemento_derivacion, dibujar_espaciador, dibujar_espacio_derivacion, dibujar_puerto_entrada, dibujar_puerto_salida
from pytc2.general import print_latex, print_subtitle
from pytc2.cuadripolos import TabcdY_s, TabcdZ_s, Z2Tabcd_s

from schemdraw.elements import  Resistor, ResistorIEC, Capacitor, Inductor, Line, Dot, Gap, Arrow
import sympy as sp

# Za, Zb = sp.symbols('Za, Zb', complex=True)
# s = sp.symbols('s', complex=True)

# Za = s
# Zb = 1/s

# T_RG = TabcdZ_s(1)
# T_RL = TabcdY_s(1)
# T_lattice = Z2Tabcd_s(sp.Matrix([[(Za+Zb)/2, (Zb-Za)/2], [(Zb-Za)/2, (Za+Zb)/2]]))

# T_tot = T_RG * T_lattice * T_RL
# Vtransf = sp.simplify(sp.expand( 1/T_tot[0,0]))
# display(Vtransf)


# Yz = 1/Z * sp.Matrix([[1, -1], [-1, 1]])

# print_subtitle('Impedancia en serie')
# display(Yz)
# dibujar_Pi(Yz)

# display(Yz[0,0] + Yz[0,1])


from pytc2.cuadripolos import *
from pytc2.general import print_latex, print_subtitle, a_equal_b_latex_s
import sympy as sp
import os

y11, y12, y21, y22 = sp.symbols('y11, y12, y21, y22', complex=True)
z11, z12, z21, z22 = sp.symbols('z11, z12, z21, z22', complex=True)
A, B, C, D = sp.symbols('A, B, C, D', complex=True)
Ai, Bi, Ci, Di = sp.symbols('Ai, Bi, Ci, Di', complex=True)
h11, h12, h21, h22 = sp.symbols('h11, h12, h21, h22', complex=True)
g11, g12, g21, g22 = sp.symbols('g11, g12, g21, g22', complex=True)
v1, v2, i1, i2 = sp.symbols('v1, v2, i1, i2', complex=True)

# Parámetros Z (impedancia - circ. abierto)
ZZ = sp.Matrix([[z11, z12], [z21, z22]])
# vars. dependientes
vv = sp.Matrix([[v1], [v2]])
# vars. INdependientes
ii = sp.Matrix([[i1], [i2]])

# Parámetros Y (admitancia - corto circ.)
YY = sp.Matrix([[y11, y12], [y21, y22]])
# vars. dependientes
# ii = sp.Matrix([[i1], [i2]])
# vars. INdependientes
# vv = sp.Matrix([[v1], [v2]])

# Parámetros H (híbridos h)
HH = sp.Matrix([[h11, h12], [h21, h22]])
# vars. dependientes
h_dep = sp.Matrix([[v1], [i2]])
# vars. INdependientes
h_ind = sp.Matrix([[i1], [v2]])

# Parámetros G (híbridos g)
GG = sp.Matrix([[g11, g12], [g21, g22]])
# vars. dependientes
g_dep = sp.Matrix([[i1], [v2]])
# vars. INdependientes
g_ind = sp.Matrix([[v1], [i2]])

# Parámetros Tabcd (Transmisión, ABCD)
TT = sp.Matrix([[A, -B], [C, -D]])
# vars. dependientes
t_dep = sp.Matrix([[v1], [i1]])
# vars. INdependientes.  (Signo negativo de corriente)
t_ind = sp.Matrix([[v2], [i2]])

# Parámetros Tdcba (Transmisión inversos, DCBA)
TTi = sp.Matrix([[Ai, Bi], [-Ci, -Di]])
# vars. dependientes
ti_dep = sp.Matrix([[v2], [i2]])
# vars. INdependientes. (Signo negativo de corriente)
ti_ind = sp.Matrix([[v1], [i1]])

s11, s12, s21, s22 = sp.symbols('s11, s12, s21, s22', complex=True)
ts11, ts12, ts21, ts22 = sp.symbols('t11, t12, t21, t22', complex=True)

# ondas normalizadas de tensión
a1, a2, b1, b2 = sp.symbols('a1, a2, b1, b2', complex=True)

# impedancia característica
Zo = sp.symbols('Zo', complex=False)

# Parámetros dispersión (scattering - S)
Spar = sp.Matrix([[s11, s12], [s21, s22]])
# vars. dependientes
bb = sp.Matrix([[b1], [b2]])
# vars. INdependientes
aa = sp.Matrix([[a1], [a2]])

# Parámetros transmisión dispersión (scattering transfer param.)
TSpar = sp.Matrix([[ts11, ts12], [ts21, ts22]])
# vars. dependientes
ts1 = sp.Matrix([[a1], [b1]])
# vars. INdependientes
ts2 = sp.Matrix([[b2], [a2]])

Tabcd_proxy_model = { 'model_name': 'T_{ABCD}', 'matrix': TT, 'dep_var': t_dep, 'indep_var':t_ind, 'neg_i2_current': True }
# Diccionario con la definición de cada modelo
model_dct = [ { 'model_name': 'S', 'matrix': Spar, 'proxy_matrix': S2Tabcd_s( Spar, Zo), 'dep_var': t_dep, 'indep_var':t_ind, 'neg_i2_current': True },
              { 'model_name': 'T_S', 'matrix': TSpar, 'proxy_matrix': Ts2Tabcd_s( TSpar, Zo), 'dep_var': t_dep, 'indep_var':t_ind, 'neg_i2_current': True },
              { 'model_name': 'T_{ABCD}', 'matrix': TT, 'dep_var': t_dep, 'indep_var':t_ind, 'neg_i2_current': True }, # T_ABCD
              { 'model_name': 'Z', 'matrix': ZZ, 'dep_var': vv, 'indep_var':ii },
              { 'model_name': 'Y', 'matrix': YY, 'dep_var': ii, 'indep_var':vv }
            ]


str_table = '$ \\begin{array}{ l ' + ' c'*len(model_dct) + ' }' + os.linesep 

for src_model in model_dct:
    str_table +=  ' & ' + src_model['model_name']

str_table = str_table + ' \\\\ ' + os.linesep

for dst_model in model_dct:

    str_table +=   dst_model['model_name']   + ' & '
    
    for src_model in model_dct:
        
        HH_z = Model_conversion( src_model, dst_model )

        str_table +=  sp.latex( HH_z['matrix'] )  + ' & '
        
    str_table = str_table[:-2] + ' \\\\ ' + os.linesep
    
    #print_latex( str_table )
    
str_table = str_table[:-4] + os.linesep + '\\end{array} $'

print_latex( str_table )



