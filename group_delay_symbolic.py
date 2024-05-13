#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:17:08 2024

@author: mariano
"""

import sympy as sp
from IPython.display import display
from pytc2.general import print_subtitle, print_latex, a_equal_b_latex_s

w0z, w0p, sigz, sigp, Qz, Qp, w = sp.symbols('\omega_{0z},\omega_{0p}, \sigma_z, \sigma_p, Q_z, Q_p, \omega', real = True)
j = sp.I

# T1jw = sigp/(j*w + sigp )
T1jw = (j*w + sigz )/(j*w + sigp )

phaT1jw = sp.atan( sp.simplify(sp.expand(sp.im(T1jw))) / sp.simplify(sp.expand(sp.re(T1jw))) )

DT1 = sp.simplify(sp.expand(-sp.diff(phaT1jw, w))) 

print_subtitle('Demora para un pasabajo de primer orden')

print_latex(a_equal_b_latex_s('D_{T1}(\omega=0)', sp.simplify(sp.expand(DT1.subs( {w:0} ))) ))

print_latex(a_equal_b_latex_s('D_{T1}(\omega=1)', sp.simplify(sp.expand(DT1.subs( {w:1} ))) ))


# T2jw = w0p**2/(-w**2+ j*w*w0p/Qp + w0p**2)
T2jw = (-w**2+ j*w*w0z/Qz + w0z**2)/(-w**2+ j*w*w0p/Qp + w0p**2)

phaT2jw = sp.atan( sp.simplify(sp.expand(sp.im(T2jw))) / sp.simplify(sp.expand(sp.re(T2jw))) )

DT2 = sp.simplify(sp.expand(-sp.diff(phaT2jw, w)))

print_subtitle('Demora para un SOS pasabajo')

print_latex(a_equal_b_latex_s('D_{T2}(\omega=0)', sp.simplify(sp.expand(DT2.subs( {w:0} ))) ))

print_latex(a_equal_b_latex_s('D_{T2}(\omega=1)', sp.simplify(sp.expand(DT2.subs( {w:1} ))) ))
print_latex(a_equal_b_latex_s('D_{T2}(\omega=1;\omega_0=1)', sp.simplify(sp.expand(DT2.subs( {w:1, w0p:1} ))) ))


display(DT2.subs( {w:0, Qp:2.02, w0p:0.997} ))
