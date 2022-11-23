#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:38:55 2021

@author: mariano

Matriz Admitancia Indefinida (MAI)
----------------------------------
Ejemplos de cálculo simbólico mediante MAI de una red escalera simplemente cargada.

Referencias:
------------
Cap. 9. Avendaño L. Sistemas electrónicos Analógicos: Un enfoque matricial.
"""

import sympy as sp
from splane import print_latex, calc_MAI_impedance_ij, calc_MAI_vtransf_ij_mn, calc_MAI_ztransf_ij_mn


# T puenteado cargado: red de R constante
# explicación:
'''    
+ Numeramos los polos de 0 a n=3

    0-------+--Y1----2---Y3--3---
                     |           /
                    Y2           / R
                     |           /
    1----------------+-------1----
    
'''    

# definición de puertos. El primer nodo marca el sentido positivo de la tensión. V1 = V[input_port[0]] - V[input_port[1]]
input_port = [0, 1]
output_port = [3, 1]

Y1, Y2, Y3 = sp.symbols('Y1 Y2 Y3', complex=True)
G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

#               Nodos: 0      1        2        3
Ymai = sp.Matrix([  
                    [ Y1,    0,      -Y1,      0],
                    [ 0,    Y2+G,    -Y2,     -G],
                    [ -Y1,  -Y2,    Y1+Y2+Y3, -Y3],
                    [ 0,    -G,      -Y3,      Y3+G ]
                 ])

s = sp.symbols('s ', complex=True)

# Butter de 3er orden simplemente cargado
Ymai = Ymai.subs(Y1, 1/s/sp.Rational('3/2'))
Ymai = Ymai.subs(Y3, 1/s/sp.Rational('1/2'))
Ymai = Ymai.subs(Y2, s*sp.Rational('4/3'))

# Butter de 3er orden doblemente cargado
# Ymai = Ymai.subs(Y1, 1/s/sp.Rational('1'))
# Ymai = Ymai.subs(Y3, 1/s/sp.Rational('1'))
# Ymai = Ymai.subs(Y2, s*sp.Rational('2'))
Ymai = Ymai.subs(G, sp.Rational('1'))

# con_detalles = False
con_detalles = True

# Calculo la Z en el puerto de entrada a partir de la MAI
Zmai = calc_MAI_impedance_ij(Ymai, input_port[0], input_port[1], verbose=con_detalles)

print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y3' )

Zmai_sym = Zmai.subs(Y3, Y1)
print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai_sym) )

print('Transferencia de tensión:')
Vmai = calc_MAI_vtransf_ij_mn(Ymai, output_port[0], output_port[1], input_port[0], input_port[1], verbose=con_detalles)
Vmai_sym = sp.simplify(Vmai.subs(Y3, Y1))

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y3' )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai_sym) )

