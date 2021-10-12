#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
from splane import print_latex
import numpy as np

# Resolución simbólica

s, z = sp.symbols('s z', complex=True)
k, fs, Q, Om = sp.symbols('k fs Q Om', real=True, positive=True)

Ts = 1/(s**2+s/Q+1)
fz = k * (z-1)/(z+1)

Tz = sp.collect(sp.simplify(sp.expand(Ts.subs(s, fz))), z)

display(Tz)
 

