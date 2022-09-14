#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de síntesis de func. de excitación mediante Foster derivación o serie.

@author: mariano
"""

import sympy as sp
import splane as tc2


def to_latex( unsimbolo ):
    
    return('$'+ sp.latex(unsimbolo)+ '$')


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)

# Implementaremos Imm mediante Foster
k0, koo, ki = tc2.foster(Imm)

# Tratamos a nuestra función imitancia como una Z
tc2.dibujar_foster_serie(k0, koo, ki, z_exc = Imm)

# Tratamos a nuestra función imitancia como una Y
tc2.dibujar_foster_derivacion(k0, koo, ki, y_exc = Imm)


