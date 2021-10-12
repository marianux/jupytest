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
from splane import print_latex, calc_MAI_impedance_ij, calc_MAI_vtransf_ij_mn, calc_MAI_ztransf_ij_mn
import numpy as np

# T puenteado cargado: red de R constante
# explicación:
'''    
+ Numeramos los polos de 0 a n=3

    0-------0---------0
            |         |      
            Y1        Y2
            |         |
            2         3
            |         |      
            Y3        Y4
            |         |
    1-------1---------1
    
'''    

# definición de puertos. El primer nodo marca el sentido positivo de la tensión. V1 = V[input_port[0]] - V[input_port[1]]
input_port = [0, 1]
output_port = [2, 3]

# nodos a contraer para que la MAI se convierta en matriz de admitancias:
nodos_a_suprimir = np.sort(np.unique([input_port[1], output_port[1]]))


Y1, Y3, Y4, Y2 = sp.symbols('Y1 Y3 Y4 Y2', complex=True)
# G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

#               Nodos: 0      1        2        3
Ymai = sp.Matrix([  
                    [ Y1+Y2,   0,      -Y1,     -Y2],
                    [ 0,      Y3+Y4,   -Y3,     -Y4],
                    [ -Y1,    -Y3,     Y1+Y3,    0],
                    [ -Y2,    -Y4,       0,     Y2+Y4]
                 ])

# con_detalles = False
con_detalles = True

# Calculo la Z en el puerto de entrada a partir de la MAI
Zmai = calc_MAI_impedance_ij(Ymai, input_port[0], input_port[1], verbose=con_detalles)

print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y4' )

Zmai_sym = Zmai.subs(Y4, Y1)
print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai_sym) )

print('Transferencia de tensión:')
Vmai = calc_MAI_vtransf_ij_mn(Ymai, output_port[0], output_port[1], input_port[0], input_port[1], verbose=con_detalles)
Vmai_sym = sp.simplify(Vmai.subs(Y4, Y1))

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y4' )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai_sym) )

## Parámetros Y Lattice

Y11 = sp.simplify(1/(1/(Y1+Y2)+1/(Y3+Y4)))
Y22 = sp.simplify(1/(1/(Y1+Y3)+1/(Y2+Y4)))
Ya = Y1 + Y2
Yb = Y3 + Y4

Y12 = sp.simplify(Ya*Y3/(Ya+Yb) - Yb*Y1/(Ya+Yb))

Ylat = sp.Matrix([ [ Y11, Y12 ], [ Y12, Y22 ] ])
Ylat_sym = sp.simplify(Ylat.subs(Y3, Y2).subs(Y4, Y1))

display(Ylat)
display(Ylat_sym)
