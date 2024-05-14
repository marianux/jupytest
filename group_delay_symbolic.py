#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:17:08 2024

@author: mariano
"""

import sympy as sp
from IPython.display import display
from pytc2.general import print_console_subtitle, print_latex, a_equal_b_latex_s
from pytc2.sistemas_lineales import pretty_print_bicuad_omegayq, s, tf2sos_analog, pretty_print_SOS, analyze_sys

w0z, w0p, sigz, sigp, Qz, Qp, w = sp.symbols('\omega_{0z},\omega_{0p}, \sigma_z, \sigma_p, Q_z, Q_p, \omega', real = True)
j = sp.I

#%% bilineal

T1s =  sigp/(s + sigp )
# T1s = (s + sigz )/(s + sigp )

print_latex(a_equal_b_latex_s('T_1(s)', T1s  ))

T1jw = T1s.subs( {s:j*w} )
# T1jw = sigp/(j*w + sigp )
# T1jw = (j*w + sigz )/(j*w + sigp )

print_latex(a_equal_b_latex_s('T_1(j\omega)', T1jw  ))

phaT1jw = sp.atan( sp.simplify(sp.expand(sp.im(T1jw))) / sp.simplify(sp.expand(sp.re(T1jw))) )

DT1 = sp.simplify(sp.expand(-sp.diff(phaT1jw, w))) 

# print_subtitle('Demora para un pasabajo de primer orden')
print_console_subtitle('Demora para un pasabajo de primer orden')

print_latex(a_equal_b_latex_s('D_{T1}(\omega=0)', sp.simplify(sp.expand(DT1.subs( {w:0} ))) ))

# print_latex(a_equal_b_latex_s('D_{T1}(\omega=1)', sp.simplify(sp.expand(DT1.subs( {w:sigp} ))) ))

#%% bicuad

T2s = w0p**2/(s**2+ s*w0p/Qp + w0p**2)
# T2s = (s**2+ s*w0z/Qz + w0z**2)/(s**2+ s*w0p/Qp + w0p**2)

print_latex(a_equal_b_latex_s('T_2(s)', T2s  ))

T2jw = T2s.subs( {s:j*w} )
# T2jw = w0p**2/(-w**2+ j*w*w0p/Qp + w0p**2)
# T2jw = (-w**2+ j*w*w0z/Qz + w0z**2)/(-w**2+ j*w*w0p/Qp + w0p**2)

print_latex(a_equal_b_latex_s('T_2(j\omega)', T2jw  ))

phaT2jw = sp.atan( sp.simplify(sp.expand(sp.im(T2jw))) / sp.simplify(sp.expand(sp.re(T2jw))) )

DT2 = sp.simplify(sp.expand(-sp.diff(phaT2jw, w)))

# print_subtitle('Demora para un SOS pasabajo')
print_console_subtitle('Demora para un SOS pasabajo')

print_latex(a_equal_b_latex_s('D_{T2}(\omega=0)', sp.simplify(sp.expand(DT2.subs( {w:0} ))) ))

print_latex(a_equal_b_latex_s('D_{T2}(\omega=\omega_{0p})', sp.simplify(sp.expand(DT2.subs( {w:w0p} ))) ))
# print_latex(a_equal_b_latex_s('D_{T2}(\omega=\omega_{0p};\omega_{0p}=1)', sp.simplify(sp.expand(DT2.subs( {w:1, w0p:1} ))) ))


# display(DT2.subs( {w:0, Qp:2.02, w0p:0.997} ))

#%% bessel

import scipy.signal as sig
import matplotlib.pyplot as plt

nn = 3

print_console_subtitle('Bessel pasabajo de orden {:d}'.format(nn))

z,p,k = sig.besselap(nn, norm='delay')

num, den = sig.zpk2tf( z,p,k)

sos = tf2sos_analog(num, den)

pretty_print_SOS(sos, mode='omegayq')

analyze_sys(sos)
# analyze_sys(sig.TransferFunction(num, den)) 

