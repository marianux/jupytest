#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:35:49 2024

@author: mariano
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Ahora importamos las funciones de PyTC2

from pytc2.remociones import remover_polo_dc, remover_polo_infinito, remover_polo_jw, isFRP, modsq2mod_s, trim_func_s

from pytc2.general import print_latex, print_subtitle, a_equal_b_latex_s
from IPython.display import display,  Markdown

# Importante importar símbolos de variables 
from pytc2.general import s, w


# Resolución simbólica

# Q de la transformación
Q_bp = 5

# nucleo de transformación pasabanda
Kbp = Q_bp * (s**2 + 1) / s


T2proto = 1/(s**2 + s + 1)

H2bp = sp.simplify(sp.expand(T2proto.subs(s, Kbp)))

print_latex(a_equal_b_latex_s('H_bp', H2bp.evalf(4)))

def plt_params(S21sq, S11sq ):
        
    # Convertir las funciones simbólicas a funciones numéricas para evaluar
    mod_S21_squared_func = sp.lambdify(w, S21sq.subs(s, sp.I*w), modules='numpy')
    mod_S11_squared_func = sp.lambdify(w, S11sq.subs(s, sp.I*w), modules='numpy')
    
    # Crear un rango de frecuencias para graficar
    frequencies = np.logspace(-1, 1, 10000)
    
    # Evaluar las funciones en el rango
    S21_vals = mod_S21_squared_func(frequencies)
    S11_vals = mod_S11_squared_func(frequencies)
    
    # Convertir a decibelios (10 * log10(valor))
    S21_dB = 10 * np.log10(S21_vals)
    S11_dB = 10 * np.log10(S11_vals)
    
    # Graficar en escala log-log
    plt.figure(figsize=(8, 5))
    plt.semilogx(frequencies, S21_dB, label=r'$|S_{21}|^2$ (dB)', color='blue')
    plt.semilogx(frequencies, S11_dB, label=r'$|S_{11}|^2$ (dB)', color='red', linestyle='--')
    plt.xlabel('Frecuencia (f) [Hz]')
    plt.ylabel('Módulo al cuadrado [dB]')
    plt.title('Complementariedad de |S21|^2 y |S11|^2 en escala log-log')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()


#%%

# Sea la siguiente función de excitación
# S21 = (s**2 *0.04) /(s**4 + s**3 *0.2828 + s**2 *2.04 + s *0.2828+  1)
# S21 = s**3/(s**3 + s**2 * 2 + s *2 + 1)
S21 = H2bp

S21sq = sp.simplify(sp.expand(S21 * S21.subs(s, -s)))


S11sq = sp.simplify(sp.expand(1 - S21sq))

# num, den = S11sq.as_numer_denom()
# num_coeffs = num.as_poly(s).all_coeffs()
# num_roots = np.roots(num_coeffs)

# conjugadas, no_conjugadas, pares_simetricos = clasificar_raices(num_roots)

# A veces puede tener sentido forzar el signo para hallar la red
# dual.
S11_sign = 1
# s11_sign = -1

S11 = S11_sign * modsq2mod_s(S11sq)

plt_params(S21, S11sq )
 

#%%

num, den = S11.as_numer_denom()

kk = sp.poly(num,s).LC()
ceros = sp.solveset(num, s, domain=sp.S.Complexes)

num_forzado_a_jw = sp.Rational(1)
for cero in ceros:
    
    if( sp.im(cero).is_zero ):
        num_forzado_a_jw = num_forzado_a_jw * s    
    elif(sp.im(cero) > 0):
        modulo = sp.Abs(cero)
        num_forzado_a_jw = num_forzado_a_jw * (s**2 + modulo**2)

num = kk * sp.simplify(sp.expand(num_forzado_a_jw))

S11 = num/den

Z1 = sp.simplify(sp.expand(den + num) / sp.expand(den - num))

print_latex(a_equal_b_latex_s('S_{21}', S21))
print_latex(a_equal_b_latex_s('S_{11}', S11.evalf(4)))
print_latex(a_equal_b_latex_s('Z_1', Z1.evalf(4)))

print_subtitle('Corroboramos el comportamiento en el centro de la banda de paso')
print_latex(a_equal_b_latex_s('Z_1(\omega = 1)', sp.simplify(sp.expand(sp.Abs(Z1.subs(s, sp.I*1)))).evalf(4) ))

