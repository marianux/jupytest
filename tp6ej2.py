#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de síntesis de func. de excitación mediante Foster derivación o serie.

@author: mariano
"""

import sympy as sp
import splane as tc2


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
YY = (s**5 + 6*s**3 + 8*s)/(s**4 + 4*s**2 + 3)

k0, imm_cauer_0, rem = tc2.cauer_LC(1/YY, remover_en_inf=False)

if rem.is_zero:
    
    print('Cauer 2: síntesis exitosa:')
    tc2.print_latex( r'$' + sp.latex(YY) + r'=' + sp.latex(imm_cauer_0) + r'$' )

    # Tratamos a nuestra función inmitancia como una Z
    tc2.dibujar_cauer_LC(k0, z_exc = 1/YY)
    
    # Tratamos a nuestra función inmitancia como una Y
    tc2.dibujar_cauer_LC(k0, y_exc = YY)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_0)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)
