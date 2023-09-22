#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 1

@author: mariano
"""

import sympy as sp
from pytc2.sintesis_dipolo import foster, cauer_LC
from pytc2.dibujar import dibujar_foster_serie, dibujar_foster_derivacion, dibujar_cauer_LC
from pytc2.general import print_latex


# from schemdraw import Drawing
# from schemdraw.elements import  Resistor


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2+3)*(s**2+1)/(s*(s**2+2))

# a) ZZ según Foster derivación

# Implementaremos Imm mediante Foster
k0, koo, ki = foster(ZZ)

dibujar_foster_serie(k0, koo, ki, z_exc = ZZ)

# Implementaremos Imm mediante Foster
k0, koo, ki = foster(1/ZZ)

dibujar_foster_derivacion(k0, koo, ki, y_exc = 1/ZZ)

# b) ZZ según Cauer1 (removiendo en oo) 

koo, imm_cauer_oo, rem = cauer_LC(ZZ, remover_en_inf=True)

print_latex( r'$' + sp.latex(ZZ) + r'=' + sp.latex(imm_cauer_oo) + r'$' )

# Tratamos a nuestra función inmitancia como una Z
dibujar_cauer_LC(koo, z_exc = imm_cauer_oo)

# b) ZZ según Cauer2 (removiendo en 0) 
k0, imm_cauer_0, rem = cauer_LC(ZZ, remover_en_inf=False)

print_latex( r'$' + sp.latex(ZZ) + r'=' + sp.latex(imm_cauer_0) + r'$' )

# Tratamos a nuestra función inmitancia como una Z
dibujar_cauer_LC(k0, z_exc = imm_cauer_0)
    


