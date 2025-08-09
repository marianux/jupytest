#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:15:12 2025

@author: mariano
"""

import numpy as np
import sympy as sp
from pytc2.sistemas_lineales import parametrize_sos
from pytc2.general import s, print_latex, print_subtitle, a_equal_b_latex_s

from pytc2.cuadripolos import smna
from IPython.display import display, Markdown


fileName_asc = '/home/mariano/mariano/Docencia/TC2/scripts/pytc2/docs/notebooks/schematics/doble_sintonizado.asc'

# symbolic MNA
equ_smna, extra_results = smna(fileName_asc, 
                               bAplicarValoresComponentes = True, 
                               bAplicarParametros = False)

print_subtitle('Ecuación MNA')

display(equ_smna)


u1 = sp.solve(equ_smna, extra_results['X'])

H = u1[extra_results['v_out']] / u1[extra_results['v_in']]

H = sp.simplify(sp.expand(H))

num, den = sp.fraction(H)

den = den.subs({r:1, k: 0.1})

f1, f2 = sp.factor(den).as_ordered_factors()

sp.roots(den, s)

L, C = sp.symbols('L,C', real = True, positive = True)

r = extra_results['comp_name_values']['R1']
l1 = extra_results['comp_name_values']['L1']
l2 = extra_results['comp_name_values']['L2']
c1 = extra_results['comp_name_values']['C1']
c2 = extra_results['comp_name_values']['C2']
k = extra_results['comp_name_values']['K']


H0 = H.subs({l1:L, l2:L, c1:C, c2:C})

num, den = sp.fraction(H0)

f1, f2 = sp.factor(den).as_ordered_factors()

f11, f21 = sp.roots(f1).as_ordered_factors()

H1 = H.subs({l1:1, l2:1, c1:1, c2:1, r:1})

num, den = sp.fraction(H1)

sp.roots(den, s)

display(H0)

#%% Simulación numérica

from pytc2.sistemas_lineales import analyze_sys, pzmap, bodePlot, pretty_print_bicuad_omegayq, tf2sos_analog, pretty_print_SOS
import scipy.signal as sig
import matplotlib.pyplot as plt


kk = np.linspace(0.001, 0.9, num=20)
# kk = np.logspace(-3, np.log10(0.8), num=20)

plt.close('all')
all_sys = []
filter_names = []

# this_axes = plt.axes()

for this_k in kk:
    
    this_tf = sig.TransferFunction([this_k,0,0,0],[(1-this_k**2), 2,3,2,1])

    all_sys.append(this_tf)

    this_label = f'k={this_k:3.3}'
    filter_names.append(this_label)
    pzmap(this_tf, annotations=False, filter_description=this_label, fig_id=2)
    bodePlot(this_tf, filter_description=this_label, fig_id=1)
    
    
# analyze_sys( all_sys, filter_names,  )
