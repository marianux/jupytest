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
import numpy as np

# Lattice simétrico: red de R constante
# explicación:
'''    
+ Numeramos los polos de 0 a n=3

    0-------0---------0
            |         |      
            Y1        Y2
            |         |
            1-----G---2
            |         |      
            Y2        Y1
            |         |
    3-------3---------3
    
'''    

# definición de puertos. El primer nodo marca el sentido positivo de la tensión. V1 = V[input_port[0]] - V[input_port[1]]
input_port = [0, 3]
output_port = [2, 1]

# nodos a contraer para que la MAI se convierta en matriz de admitancias:
nodos_a_suprimir = np.sort(np.unique([input_port[1], output_port[1]]))


Y1, Y2 = sp.symbols('Y1 Y2', complex=True)
G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

#               Nodos: 0      1        2        3
Ymai = sp.Matrix([  
                    [ Y1+Y2,  -Y1,      -Y2,         0],
                    [ -Y1,     Y2+Y1+G, -G,        -Y2],
                    [ -Y2,    -G,       Y1+Y2+G,   -Y1],
                    [   0,    -Y2,      -Y1,     Y2+Y1]
                 ])

con_detalles = False
# con_detalles = True

print('\nRed lattice:\n')

# Calculo la Z en el puerto de entrada a partir de la MAI
Zmai = calc_MAI_impedance_ij(Ymai, input_port[0], input_port[1], verbose=con_detalles)
Zmai = sp.simplify(sp.expand(Zmai.subs(Y1*Y2, G**2)))

print_latex( r'Z_{{ {:d}{:d} }} = '.format(input_port[0], input_port[1]) +  sp.latex(Zmai) )

print('Transferencia de tensión:')
Vmai = calc_MAI_vtransf_ij_mn(Ymai, output_port[0], output_port[1], input_port[0], input_port[1], verbose=con_detalles)
Vmai = sp.simplify(sp.expand(Vmai.subs(Y1*Y2, G**2)))

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai) )
