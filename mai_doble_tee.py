#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:40:23 2025

@author: mariano

Ejemplo de análisis de red doble Tee RC en configuración notch y verificación 
via MAI y cuadripolos.
"""

import sympy as sp

from pytc2.cuadripolos import calc_MAI_impedance_ij, calc_MAI_vtransf_ij_mn, \
                                  calc_MAI_ztransf_ij_mn, Y2Tabcd_s, smna
from pytc2.general import print_latex, s, print_subtitle, a_equal_b_latex_s

from pytc2.dibujar import dibujar_puerto_salida, dibujar_puerto_entrada
from schemdraw import Drawing
from schemdraw.elements import ResistorIEC, Line, Resistor, Capacitor, Dot
from IPython.display import display


# dibujamos la red T-puenteada
with Drawing() as d:
    d.config(fontsize=16, unit=4)
    d = dibujar_puerto_entrada(d, port_name = '' )
    d.push()
    d += Dot().label('0', loc='bottom')
    d += Line().length(d.unit*.4).up()
    d += (zb_up := Capacitor().right().label('C'))
    d += Dot().label('2')
    d += Capacitor().right().label('C')   
    d += Line().length(d.unit*.4).down()
    d += Dot().label('3', loc='bottom')
    d.pop()
    d += Line().length(d.unit*.2).right()
    d += Resistor().right().label('G')
    d += Dot().label('1', loc='top')
    d.push()
    d += (za_down := Resistor().right().label('G'))
    d = dibujar_puerto_salida(d, port_name = '' )
    d.pop()
    d += Capacitor().label('2C', loc='bottom').down()
    d += Dot().label('4', loc='bottom')
    y_down  = d.here[1]
    d += Resistor().down().label('2G').idot().dot().endpoints( zb_up.end, [zb_up.end[0], y_down ])
    d += Line().endpoints( [zb_up.start[0], y_down ], [za_down.end[0], y_down ])


display(d)

G, C = sp.symbols('G, C', complex=False)
# G = sp.symbols('G', real=True, positive=True)

# Armo la MAI

print_subtitle('Análisis via matriz admitancia indefinida')

#               Nodos: 0         1         2           3         4
Ymai = sp.Matrix([  
                    [ G+s*C,   -G,       -s*C,         0,       0 ],
                    [ -G,    2*(G+s*C),    0,        -G,      -2*s*C],
                    [ -s*C,     0,      2*(s*C+G),  -s*C,      -2*G],
                    [ 0,       -G,       -s*C,       G+s*C,       0 ],
                    [ 0,       -2*s*C,   -2*G,        0,         2*(G+s*C) ]
                 ])

print_latex( a_equal_b_latex_s('Y_{MAI}', Ymai ))

subs_dict = {G:1, C:1/2}

con_detalles = False
# con_detalles = True

InPort = [0, 4]
OutPort = [3, 4]

# Calculo la Z en el puerto de entrada a partir de la MAI
Zin = calc_MAI_impedance_ij(Ymai, InPort, verbose=con_detalles)

Zmai = calc_MAI_ztransf_ij_mn(Ymai, InPort, OutPort, verbose=con_detalles)

Vmai = calc_MAI_vtransf_ij_mn(Ymai, InPort, OutPort, verbose=con_detalles)


print_latex( r'Z_{{ {:d}{:d} }} = '.format(InPort[0], InPort[1]) +  sp.latex(Zin.subs(subs_dict)) )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(OutPort[0], OutPort[1], InPort[0], InPort[1]) +  sp.latex(Vmai) + ' = ' + sp.latex(Vmai.subs(subs_dict)) )

print_latex( r'Z^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(OutPort[0], OutPort[1], InPort[0], InPort[1]) + sp.latex(Zmai) + ' = ' + sp.latex(Zmai.subs(subs_dict)) )


# %%
print_subtitle('Análisis via Cuadripolos')



Ztee1 = sp.Matrix([[ 1/(2*G) + 1/(s*C) , 1/(2*G)], [1/(2*G), 1/(2*G) + 1/(s*C)]])
Ztee2 = sp.Matrix([[ 1/G + 1/(s*2*C) , 1/(s*2*C)], [1/(s*2*C), 1/G + 1/(s*2*C)]])

Ttt = Y2Tabcd_s( Ztee1**-1 + Ztee2**-1 )

print('T_{tot} = Y2Tabcd_s( Ztee1**-1 + Ztee2**-1 )')

print_latex( a_equal_b_latex_s('T_{tot}', Ttt))

Zin_tp = Ttt[0,0] / Ttt[1,0]
V_tp = 1/ Ttt[0,0]
Z_tp = 1/ Ttt[1,0]

print_latex( r'Z_{{ {:d}{:d} }} = '.format(InPort[0], InPort[1]) +  sp.latex(Zin_tp.subs(subs_dict)) )

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(OutPort[0], OutPort[1], InPort[0], InPort[1]) +  sp.latex(V_tp) + ' = ' + sp.latex(V_tp.subs(subs_dict)) )

print_latex( r'Z^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(OutPort[0], OutPort[1], InPort[0], InPort[1]) + sp.latex(Z_tp) + ' = ' + sp.latex(Z_tp.subs(subs_dict)) )



# %%

print_subtitle('Análisis via Modified Nodal Analysis (MNA)')

fileName_asc = '/home/mariano/Desktop/Enlace hacia spice/doble_tee_notch.asc'

# symbolic MNA
equ_smna, extra_results = smna(fileName_asc, 
                               bAplicarValoresComponentes = True, 
                               bAplicarParametros = True)

print_subtitle('Ecuación MNA')

display(equ_smna)

u1 = sp.solve(equ_smna, extra_results['X'])

H = u1[extra_results['v_out']] / u1[extra_results['v_in']]

H = sp.simplify(sp.expand(H))

# num, den = sp.fraction(H)

print_latex( r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = '.format(OutPort[0], OutPort[1], InPort[0], InPort[1]) +  sp.latex(H) )
