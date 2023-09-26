#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Ejemplo de síntesis de func. de excitación NO DISIPATIVAS mediante los 
métodos de Cauer.

@author: mariano
"""

import sympy as sp
from IPython.display import display

from pytc2.sintesis_dipolo import cauer_LC
from pytc2.dibujar import dibujar_cauer_LC
from pytc2.general import print_latex, print_console_subtitle, a_equal_b_latex_s


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)

print_console_subtitle('Cauer 1: se remueve en oo')

# Implementaremos Imm mediante Cauer 1 o remociones continuas en infinito
koo, imm_cauer_oo, rem = cauer_LC(Imm, remover_en_inf=True)

if rem.is_zero:

    print('Síntesis exitosa!')

    print_latex(a_equal_b_latex_s('Imm(s)', imm_cauer_oo ))

    # Tratamos a nuestra función inmitancia como una Z
    dibujar_cauer_LC(koo, z_exc = imm_cauer_oo)
    
    # Tratamos a nuestra función inmitancia como una Y
    dibujar_cauer_LC(koo, y_exc = imm_cauer_oo)

else:
    
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_oo)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)
    
        
print_console_subtitle('Cauer 2: se remueve en 0')

# Implementaremos Imm mediante Cauer 2 o remociones continuas en cero
k0, imm_cauer_0, rem = cauer_LC(Imm, remover_en_inf=False)

if rem.is_zero:
    
    print('Síntesis exitosa!')

    print_latex(a_equal_b_latex_s('Imm(s)', imm_cauer_0 ))

    # Tratamos a nuestra función inmitancia como una Z
    dibujar_cauer_LC(k0, z_exc = imm_cauer_0)
    
    # Tratamos a nuestra función inmitancia como una Y
    dibujar_cauer_LC(k0, y_exc = imm_cauer_0)

else:
    print('Hubo algún problema con la síntesis. Se pudo sintetizar:')
    
    display(imm_cauer_0)
    
    print('Quedó por sintetizar la siguiente función:')
    
    display(rem)





