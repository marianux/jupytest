#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:25:23 2022

Análisis de un inductor real modelizado mediante parámetros S

          +---R2--C--+
    --R1--+----L-----+----

@author: mariano
"""

import sympy as sp
import splane as tc2
from splane import s


R1, R2, C, L = sp.symbols('R1, R2, C, L', complex=False)

ZLreal = R1 + tc2.pp(s*L, R2 + 1/(s*C))
ZLreal = sp.simplify(sp.expand( ZLreal ))

num, den = sp.fraction(sp.simplify(sp.expand(ZLreal)))

# Para Ro = 1
S11 = tc2.simplify_n_monic( ZLreal/(ZLreal+2) )
S21 = tc2.simplify_n_monic( 2/(ZLreal+2) )

ZLmedida = ZLreal.subs([(R1, sp.Rational('1/2')), (R2, sp.Rational('1/2')), (C, sp.Rational('1/10')), (L, sp.Rational('1'))] )
S11medida = S11.subs([(R1, sp.Rational('1/2')), (R2, sp.Rational('1/2')), (C, sp.Rational('1/10')), (L, sp.Rational('1'))] )
S21medida = S21.subs([(R1, sp.Rational('1/2')), (R2, sp.Rational('1/2')), (C, sp.Rational('1/10')), (L, sp.Rational('1'))] )

S11aprox = (s**2 + 41/4 *s + 5) / (3*s**2 + 11*s + 25)
S21aprox = (2*s**2 + s + 20) / (3*s**2 + 45/4*s + 25)

ZLcalc = tc2.simplify_n_monic( 2 * (1-S21aprox)/S21aprox)
