#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:44:42 2023

@author: mariano
"""


import sympy as sp

# Ahora importamos las funciones de PyTC2

from pytc2.sintesis_dipolo import cauer_RC, foster, foster_zRC2yRC
from pytc2.dibujar import dibujar_cauer_RC_RL, dibujar_foster_derivacion, dibujar_foster_serie
from pytc2.general import print_latex, print_subtitle, expr_simb_expr, a_equal_b_latex_s
from IPython.display import display,  Markdown
from pytc2.sintesis_dipolo import cauer_RC
from pytc2.dibujar import dibujar_cauer_RC_RL

# Resolución simbólica
s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2 + 2*s + 2)/(s**2 + 1*s + 1)

print_subtitle('Impedancia $Z_{RLC}$ ')

print_latex(a_equal_b_latex_s('Z_{RLC}(s)', ZZ))


# Implementaremos FF mediante Cauer 1 o remociones continuas en infinito
koo, ZZ_cauer_oo, rem = cauer_RC(ZZ, remover_en_inf=True)

print_subtitle('Implementación escalera de $Z_{RLC}$ e $Y_{RLC}$')

print_latex(a_equal_b_latex_s(a_equal_b_latex_s('$ Z_{RLC}(s)', ZZ)[1:-1], ZZ_cauer_oo ))

# Tratamos a nuestra función inmitancia como una Z
dibujar_cauer_RC_RL(koo, z_exc = ZZ_cauer_oo)

print_latex(a_equal_b_latex_s(a_equal_b_latex_s('$ Y_{RLC}(s)', ZZ)[1:-1], ZZ_cauer_oo ))

# Tratamos a nuestra función inmitancia como una Y
dibujar_cauer_RC_RL(koo, y_exc = ZZ_cauer_oo)

