#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de síntesis de func. de excitación DISIPATIVAS mediante los métodos 
de Cauer.

@author: mariano
"""

import sympy as sp
import splane as tc2


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
Imm = (s**2 + 4*s + 3)/(s**2 + 2*s)

# Implementaremos Imm mediante Cauer 1 o remociones continuas en infinito
koo, imm_cauer_oo, rem = tc2.cauer_RC(Imm, remover_en_inf=True)

if rem.is_zero:
    
    print('Cauer 1: síntesis exitosa:')
    tc2.print_latex( r'$' + sp.latex(Imm) + r'=' + sp.latex(imm_cauer_oo) + r'$' )

    # Tratamos a nuestra función inmitancia como una Z
    tc2.dibujar_cauer_RC_RL(koo, z_exc = imm_cauer_oo)
    
    # Tratamos a nuestra función inmitancia como una Y
    tc2.dibujar_cauer_RC_RL(koo, y_exc = imm_cauer_oo)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_oo)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)


# Implementaremos Imm mediante Cauer 2 o remociones continuas en cero
k0, imm_cauer_0, rem = tc2.cauer_RC(Imm, remover_en_inf=False)

if rem.is_zero:
    
    print('Cauer 2: síntesis exitosa:')
    tc2.print_latex( r'$' + sp.latex(Imm) + r'=' + sp.latex(imm_cauer_0) + r'$' )

    # Tratamos a nuestra función inmitancia como una Z
    tc2.dibujar_cauer_RC_RL(k0, z_exc = imm_cauer_0)
    
    # Tratamos a nuestra función inmitancia como una Y
    tc2.dibujar_cauer_RC_RL(k0, y_exc = imm_cauer_0)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_oo)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)




