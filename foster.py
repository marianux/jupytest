#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de síntesis de func. de excitación mediante Foster derivación o serie.

@author: mariano
"""

import sympy as sp

from pytc2.sintesis_dipolo import foster
from pytc2.dibujar import dibujar_foster_serie, dibujar_foster_derivacion
from pytc2.general import print_latex, print_console_subtitle, a_equal_b_latex_s



# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
FF = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)

# Implementaremos FF mediante Foster
k0, koo, ki, FF_foster = foster(FF)

print_console_subtitle('Foster serie')

print_latex(a_equal_b_latex_s(a_equal_b_latex_s('Z(s)=F(s)', FF)[1:-1], FF_foster ))

# Tratamos a nuestra función imitancia como una Z
dibujar_foster_serie(k0, koo, ki, z_exc = FF)

print_console_subtitle('Foster derivación')

print_latex(a_equal_b_latex_s(a_equal_b_latex_s('Y(s)=F(s)', FF)[1:-1], FF_foster ))


# Tratamos a nuestra función imitancia como una Y
dibujar_foster_derivacion(k0, koo, ki, y_exc = FF)


