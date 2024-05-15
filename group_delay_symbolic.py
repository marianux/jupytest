#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:17:08 2024

@author: mariano
"""

import numpy as np
import sympy as sp
from IPython.display import display
from pytc2.general import print_console_subtitle, print_latex, a_equal_b_latex_s
from pytc2.sistemas_lineales import pretty_print_bicuad_omegayq, s, tf2sos_analog, pretty_print_SOS, analyze_sys, tfadd

w0z, w0p, sigz, sigp, Qz, Qp, w = sp.symbols('\omega_{0z},\omega_{0p}, \sigma_z, \sigma_p, Q_z, Q_p, \omega', real = True)
j = sp.I

#%% bilineal

# T1s =  sigp/(s + sigp )
T1s = (s + sigz )/(s + sigp )

print_latex(a_equal_b_latex_s('T_1(s)', T1s  ))

T1jw = T1s.subs( {s:j*w} )
# T1jw = sigp/(j*w + sigp )
# T1jw = (j*w + sigz )/(j*w + sigp )

print_latex(a_equal_b_latex_s('T_1(j\omega)', T1jw  ))

phaT1jw = sp.atan( sp.simplify(sp.expand(sp.im(T1jw))) / sp.simplify(sp.expand(sp.re(T1jw))) )

DT1 = sp.simplify(sp.expand(-sp.diff(phaT1jw, w))) 

print_latex(a_equal_b_latex_s('\phi_{T1}(\omega)', phaT1jw ))

print_latex(a_equal_b_latex_s('D_{T1}(\omega)', DT1 ))

# print_subtitle('Demora para un pasabajo de primer orden')
print_console_subtitle('Demora para un pasabajo de primer orden')

print_latex(a_equal_b_latex_s('D_{T1}(\omega=0)', sp.simplify(sp.expand(DT1.subs( {w:0} ))) ))

# print_latex(a_equal_b_latex_s('D_{T1}(\omega=1)', sp.simplify(sp.expand(DT1.subs( {w:sigp} ))) ))

#%% bicuad

# T2s = w0p**2/(s**2+ s*w0p/Qp + w0p**2)
# T2s = (s**2+ s*w0z/Qz + w0z**2)/(s**2+ s*w0p/Qp + w0p**2)
# T2s = sp.factor((s**2+ s*w0z/Qz + w0z**2))

# factorizando polinomios no funcionó
num_roots = sp.roots( (s**2+ s*w0z/Qz + w0z**2).as_poly(s) )
den_roots = sp.roots( (s**2+ s*w0p/Qp + w0p**2).as_poly(s) )

T2sz = sp.Rational(1.)
for root, multiplicity in num_roots.items():
    for ii in range(multiplicity):
        T2sz = T2sz * (s - root)

T2sp = sp.Rational(1.)    
for root, multiplicity in den_roots.items():
    for ii in range(multiplicity):
        T2sp = T2sp / (s - root)
        
# Analizo la demora del num y den por separado, ya que num resta demora y den suma
T2s = T2sz/T2sp 

print_latex(a_equal_b_latex_s('T_2(s)', T2s  ))

T2jw = T2s.subs( {s:j*w} )
# T2jw = w0p**2/(-w**2+ j*w*w0p/Qp + w0p**2)
# T2jw = (-w**2+ j*w*w0z/Qz + w0z**2)/(-w**2+ j*w*w0p/Qp + w0p**2)

T2zjw = T2sz.subs( {s:j*w} )
T2pjw = T2sp.subs( {s:j*w} )

print_latex(a_equal_b_latex_s('T_2(j\omega)', T2jw  ))

# phaT2jw = sp.atan( sp.simplify(sp.expand(sp.im(T2jw))) / sp.simplify(sp.expand(sp.re(T2jw))) )
phaT2zjw = sp.atan( sp.simplify(sp.expand(  sp.simplify(sp.expand(sp.im(T2zjw))) / sp.simplify(sp.expand(sp.re(T2zjw))) )) )
phaT2pjw = sp.atan( sp.simplify(sp.expand( sp.simplify(sp.expand(sp.im(T2pjw))) / sp.simplify(sp.expand(sp.re(T2pjw))) )) )
phaT2jw = phaT2zjw + phaT2pjw 

# DT2 = sp.simplify(sp.expand(-sp.diff(phaT2jw, w)))
DT2z = sp.simplify(sp.expand(-sp.diff(phaT2zjw, w)))
DT2p = sp.simplify(sp.expand(-sp.diff(phaT2pjw, w)))

DT2 = DT2z + DT2p

# print_subtitle('Demora para un SOS pasabajo')
print_console_subtitle('Demora para un SOS pasabajo')

print_latex(a_equal_b_latex_s('\phi_{T2}(\omega)', phaT2jw ))

print_latex(a_equal_b_latex_s('D_{T2}(\omega)', DT2  ))

print_latex(a_equal_b_latex_s('D_{T2}(\omega=0)', sp.simplify(sp.expand(DT2.subs( {w:0} ))) ))

print_latex(a_equal_b_latex_s('D_{T2}(\omega=\omega_{0p})', sp.simplify(sp.expand(DT2.subs( {w:w0p} ))) ))
# print_latex(a_equal_b_latex_s('D_{T2}(\omega=\omega_{0p};\omega_{0p}=1)', sp.simplify(sp.expand(DT2.subs( {w:1, w0p:1} ))) ))


# display(DT2.subs( {w:0, Qp:2.02, w0p:0.997} ))

#%% bessel

import scipy.signal as sig
import matplotlib.pyplot as plt

nn = 3

print_console_subtitle('Bessel pasabajo de orden {:d}'.format(nn))

z,p,k = sig.besselap(nn, norm='phase')

num_lp, den_lp = sig.zpk2tf( z,p,k)

sos_lp = tf2sos_analog(num_lp, den_lp)

tf_lp = sig.TransferFunction(num_lp, den_lp)

pretty_print_SOS(sos_lp, mode='omegayq')


num_hp, den_hp = sig.lp2hp(num_lp, den_lp)

sos_hp = tf2sos_analog(num_hp, den_hp )

tf_hp = sig.TransferFunction(num_hp, den_hp)

pretty_print_SOS(sos_hp, mode='omegayq')

tfXover = tfadd(tf_lp, tf_hp)
sos_xover = tf2sos_analog(tfXover.num, tfXover.den )

analyze_sys([tf_lp, tf_hp, sos_xover])

# analyze_sys(sig.TransferFunction(num, den)) 

