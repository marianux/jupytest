#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS10 ej 2

@author: mariano
"""

import sympy as sp
import numpy as np
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor, Line, Dot
import splane as tc2



# Resolución simbólica

s = sp.symbols('s ', complex=True)

ZZ = (s**2+s+1)/(s**2+2*s+5)/(s+1)

# remoción total en infinito de 1/ZZ

Y2, k1 = tc2.remover_polo_ (sigma1, yy = 1/ZZ)

C1 = sp.limit(1/s/ZZ,s, sp.oo)

# extraigo C1
Y2 = sp.factor(sp.simplify(sp.expand(1/ZZ - C1*s)))

Ginf = sp.limit(Y2, s, sp.oo)
G0 = sp.limit(Y2, s, 0)

# remuevo la menor admitancia
R1 = 1/np.min((Ginf, G0))
Y4 = sp.factor(sp.simplify(sp.expand(Y2 - 1/R1)))

L1 = sp.limit(1/s/Y4, s, sp.oo)

# extraigo L1
Z6 = sp.factor(sp.simplify(sp.expand(1/Y4 - L1*s)))

Rinf = sp.limit(Z6, s, sp.oo)
R0 = sp.limit(Z6, s, 0)

# remuevo la menor resistencia
R2 = np.min((Rinf, R0))
Z8 = sp.factor(sp.simplify(sp.expand(Z6 - R2)))

# extraigo C2
C2 = sp.limit(1/s/Z8, s, sp.oo)

Y10 = sp.factor(sp.simplify(sp.expand(1/Z8 - C2*s)))

R3 = 1/Y10


# fumada para dibujar el circuito resultante
d = Drawing(unit=2)  # unit=2 makes elements have shorter than normal leads
d += Dot()
d += Line().right()
d.push()
d += Dot()
d += (Capacitor().down().label(to_latex(C1)))
d.pop()
d += Line().right()
d.push()
d += Dot()
d += Resistor().down().label(to_latex(R1))
d.pop()
d += Inductor().right().label(to_latex(L1))
d += Resistor().right().label(to_latex(R2))
d.push()
d += Dot()
d += (Capacitor().down().label(to_latex(C2)))
d.pop()
d += Line().right()
d += Resistor().down().label(to_latex(R3))
d += Line().left()
d += Dot()
d += Line().left()
d += Line().left()
d += Dot()
d += Line().left()
d += Dot()
d += Line().left()
d += Dot()

display(d)


