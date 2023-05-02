#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:38:55 2021

@author: mariano

Matriz Admitancia Indefinida (MAI)
----------------------------------
Ejemplos de cálculo simbólico mediante MAI de una red T puenteada de R constante.

Referencias:
------------
Cap. 9. Avendaño L. Sistemas electrónicos Analógicos: Un enfoque matricial.
"""

import sympy as sp

from pytc2.cuadripolos import calc_MAI_impedance_ij, calc_MAI_vtransf_ij_mn, calc_MAI_ztransf_ij_mn
from pytc2.general import print_latex


# T puenteado cargado: red de R constante
# explicación:
'''    
+ Numeramos los polos de 0 a n=3

            |------Ya-------|
            |               |
    0-------+--G----2---G---3
                    |       |
                   Yb       G
                    |       |
    1---------------+--------
    
'''    

Ya, Yb = sp.symbols('Ya Yb', complex=True)
G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

#               Nodos: 0      1        2        3
Ymai = sp.Matrix([  
                    [ Ya+G,   0,      -G,     -Ya],
                    [ 0,     Yb+G,    -Yb,    -G],
                    [ -G,   -Yb,      2*G+Yb, -G],
                    [ -Ya,   -G,     -G,      2*G+Ya ]
                 ])

# con_detalles = False
con_detalles = True

# Calculo la Z en el puerto de entrada a partir de la MAI
Zmai = calc_MAI_impedance_ij(Ymai, 0, 1, verbose=con_detalles)

# Aplico la condición de R constante
print('si consideramos:')
print_latex( r'G^2 = Y_a . Y_b' )
print('entonces')
print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai.subs(Ya*Yb, G**2)) )

print('Transferencia de tensión:')
Vmai = calc_MAI_vtransf_ij_mn(Ymai, 3, 1, 0, 1, verbose=con_detalles)
Vmai = sp.simplify(Vmai.subs(Ya*Yb, G**2))
Vmai_Ya = sp.simplify(Vmai.subs(Yb, G**2/Ya))
Vmai_Yb = sp.simplify(Vmai.subs(Ya, G**2/Yb))

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(3, 1, 0, 1) +  sp.latex(Vmai_Ya) + ' = ' + sp.latex(Vmai_Yb) )

print('Transimpedancia:')
Zmai = calc_MAI_ztransf_ij_mn(Ymai, 3, 1, 0, 1, verbose=con_detalles)
Zmai = sp.simplify(Zmai.subs(Ya*Yb, G**2))
Zmai_Ya = sp.simplify(Zmai.subs(Yb, G**2/Ya))
Zmai_Yb = sp.simplify(Zmai.subs(Ya, G**2/Yb))
print_latex( r'Z^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(3, 1, 0, 1) + sp.latex(Zmai_Ya) + ' = ' + sp.latex(Zmai_Yb) )

