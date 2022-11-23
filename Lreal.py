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

s11 = sp.simplify(sp.expand(ZLreal/(ZLreal+2)))
s21 = sp.simplify(sp.expand((2/(ZLreal+2))))

s11_modelada = s11.subs([(R1, sp.Rational('2')), (R2, sp.Rational('2')), (C, sp.Rational('1/2')), (L, sp.Rational('10'))])
s21_modelada = s21.subs([(R1, sp.Rational('2')), (R2, sp.Rational('2')), (C, sp.Rational('1/2')), (L, sp.Rational('10'))])

# PP, QQ = s11_modelada.as_numer_denom()
# RR, QQ = s21_modelada.as_numer_denom()

# ZL_s11_calc = sp.simplify(sp.expand((2*s11_modelada/(1-s11_modelada))))
ZL_s21_calc = sp.simplify(sp.expand((2*(1-s21_modelada)/s21_modelada)))


# Síntesis guiada por topología

z2 = sp.simplify(sp.expand(ZL_s21_calc - 2))

y4, YL4 = tc2.remover_polo_dc(1/z2)






