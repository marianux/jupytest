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
ZLreal = tc2.simplify_n_monic( ZLreal )

sp.simplify(sp.expand(ZLreal ))