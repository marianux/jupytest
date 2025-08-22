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
from pytc2.cuadripolos import calc_MAI_impedance_ij, calc_MAI_vtransf_ij_mn, \
                              may2y, Y2Tabcd_s
from pytc2.general import print_latex, s, print_subtitle, a_equal_b_latex_s

import numpy as np

from pytc2.dibujar import dibujar_Pi

    
    

# T puenteado cargado: red de R constante
# explicación:
'''    
+ Numeramos los polos de 0 a n=3

    0-------0---Y2----2------2
            |         |      
            Y1        Y3
            |         |
    1-------1---------1------1
    
'''    


# definición de puertos. El primer nodo marca el sentido positivo de la tensión. V1 = V[input_port[0]] - V[input_port[1]]
input_port = [0, 1]
output_port = [2, 1]



Y1, Y2, Y3 = sp.symbols('Y1 Y2 Y3', complex=True)
# G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

#               Nodos: 0       1         2      
Ymai = sp.Matrix([  
                    [ Y1+Y2,  -Y1,      -Y2 ],
                    [ -Y1,    Y1+Y3,   -Y3  ],
                    [ -Y2,    -Y3,     Y2+Y3]
                 ])

con_detalles = False
# con_detalles = True

# Calculo la Z en el puerto de entrada a partir de la MAI
Zmai = calc_MAI_impedance_ij(Ymai, input_port, verbose=con_detalles)

print_subtitle('Análisis via MAI')

print_latex( a_equal_b_latex_s('Y_{MAI}', Ymai ))

print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y3' )

Zmai_sym = Zmai.subs(Y3, Y1)
print_latex( r'Z_{{ {:d}{:d} }} = '.format(0,1) +  sp.latex(Zmai_sym) )

print('Transferencia de tensión:')
Vmai = calc_MAI_vtransf_ij_mn(Ymai, input_port, output_port, verbose=con_detalles)
Vmai_sym = sp.simplify(Vmai.subs(Y3, Y1))

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai) )

print('si consideramos simétrica la red:')
print_latex( r'Y1 = Y3' )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(Vmai_sym) )


print_subtitle('Convertimos de MAI a Pi definido por cuadripolo Y')
print('input_port')
print(input_port)
print('output_port')
print(output_port)

# nodos a contraer para que la MAI se convierta en matriz de admitancias:
nodos_a_suprimir = [1]

YY = may2y(Ymai, nodos_a_suprimir)

print_latex( a_equal_b_latex_s('Y_{\pi}', YY ))

dibujar_Pi(YY)

Ttt = Y2Tabcd_s( YY )

print_latex( a_equal_b_latex_s('T_{\pi}', Ttt))

Zin_tp = Ttt[0,0] / Ttt[1,0]
V_tp = 1/ Ttt[0,0]
Z_tp = 1/ Ttt[1,0]

print_latex( r'Z_{{ {:d}{:d} }} = '.format(input_port[0], input_port[1]) +  sp.latex(Zin_tp) )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) +  sp.latex(V_tp)  )

print_latex( r'Z^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(output_port[0], output_port[1], input_port[0], input_port[1]) + sp.latex(Z_tp)  )

