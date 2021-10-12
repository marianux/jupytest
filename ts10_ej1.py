#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 1

@author: mariano
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor, Line, Dot


def to_latex( unsimbolo ):
    
    return('$'+ sp.latex(unsimbolo)+ '$')


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2+6*s+8)/(s**2+4*s+3)

# Halle los valores de los componentes de la topología resultante.

# remoción parcial para que el siguiente tanque R1-C1 resuenen a 6 r/s

sigma1 = 6
Ra = sp.simplify(ZZ.subs(s, -sigma1))

Z2 = sp.factor(sp.simplify(sp.expand(ZZ - Ra)))

k1 = sp.limit((s+6)/s/Z2,s,-sigma1)

R1 = 1/k1
C1 = k1/sigma1

Y4 = sp.factor(sp.simplify(sp.expand(1/Z2 - k1*s/(s+6))))

sigma2 = sp.Rational('7/2')

Rb = sp.expand((1/Y4).subs(s, -sigma2 ))

Z6 = sp.factor(sp.simplify(sp.expand(1/Y4 - Rb)))

k2 = sp.limit((s+7/2)/s/Z6,s, -sigma2 )

R2 = 1/k2
C2 = k2/sigma2

Rc = 1/sp.factor(sp.simplify(sp.expand(1/Z6 - k2*s/(s+sigma2))))


# fumada para dibujar el circuito resultante
d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads
d += Dot()
d += (dr_RA := Resistor().label(to_latex(Ra)))
d.push()
d += Dot()
d += Resistor().right().label(to_latex(Rb))
d += Dot()
d += Line().right()
d += Line().down()
d += Resistor().down().label(to_latex(Rc), loc='bottom')
d += Line().left()
d += Dot()
d += Resistor().up().label(to_latex(R2), loc='bottom')
d += Capacitor().up().label(to_latex(C2), loc='bottom')
d.pop()
d += Capacitor().down().label(to_latex(C1), loc='bottom')
d += Resistor().down().label(to_latex(R1), loc='bottom')
d += Dot()
d += Line().right()
d += Line().left().tox(dr_RA.start)
d += Dot()

# funciona solo en modo interactivo ... no sé por qué
# d.draw() 

print('Ejecutar d.draw() para visualizar el circuito')
