#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de síntesis de func. de excitación DISIPATIVAS mediante los métodos 
de Cauer.

@author: mariano
"""

import sympy as sp

from pytc2.sintesis_dipolo import cauer_RC, foster, foster_zRC2yRC
from pytc2.dibujar import dibujar_cauer_RC_RL, dibujar_foster_derivacion, dibujar_foster_serie
from pytc2.general import print_latex
from IPython.display import display


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
# ZRC - YRL
Imm = (s**2 + 4*s + 3)/(s**2 + 2*s)
# k0, koo, ki_wi, kk, FF_foster = foster(Imm)

# YRC - ZRL
# Imm = 2*(s**2 + 4*s + 3)/(s**2 + 8*s + 12)
# k0, koo, ki_wi, kk, YRC_foster = foster(Imm/s)
# k0, koo, ki_wi, kk, YRC_foster = foster_zRC2yRC(k0, koo, ki_wi, kk, YRC_foster)

# dibujar_foster_serie(k0 = k0, koo = koo, ki = ki_wi, kk = kk, z_exc = FF_foster)


# Implementaremos Imm mediante Cauer 1 o remociones continuas en infinito
koo, imm_cauer_oo, rem = cauer_RC(Imm, remover_en_inf=True)

if rem.is_zero:
    
    print('Cauer 1: síntesis exitosa:')
    print_latex( r'$' + sp.latex(Imm) + r'=' + sp.latex(imm_cauer_oo) + r'$' )

    # Tratamos a nuestra función inmitancia como una Z
    dibujar_cauer_RC_RL(koo, z_exc = imm_cauer_oo)
    
    # Tratamos a nuestra función inmitancia como una Y
    dibujar_cauer_RC_RL(koo, y_exc = imm_cauer_oo)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_oo)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)


# Implementaremos Imm mediante Cauer 2 o remociones continuas en cero
k0, imm_cauer_0, rem = cauer_RC(Imm, remover_en_inf=False)

if rem.is_zero:
    
    print('Cauer 2: síntesis exitosa:')
    print_latex( r'$' + sp.latex(Imm) + r'=' + sp.latex(imm_cauer_0) + r'$' )

    # Tratamos a nuestra función inmitancia como una Z
    dibujar_cauer_RC_RL(k0, z_exc = imm_cauer_0)
    
    # Tratamos a nuestra función inmitancia como una Y
    dibujar_cauer_RC_RL(k0, y_exc = imm_cauer_0)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_oo)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)




