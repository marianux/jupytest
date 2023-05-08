#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:06:52 2023

@author: mariano
"""

from pytc2.dibujar import dibujar_Pi, dibujar_Tee, dibujar_elemento_serie, dibujar_elemento_derivacion, dibujar_espaciador, dibujar_espacio_derivacion, dibujar_puerto_entrada, dibujar_puerto_salida
from pytc2.general import print_latex, print_subtitle
from schemdraw.elements import  Resistor, ResistorIEC, Capacitor, Inductor, Line, Dot, Gap, Arrow
import sympy as sp

# Definimos la matriz Yz

Y, Z = sp.symbols('Y, Z', complex=True)

Yz = 1/Z * sp.Matrix([[1, -1], [-1, 1]])

print_subtitle('Impedancia en serie')
display(Yz)
dibujar_Pi(Yz)

display(Yz[0,0] + Yz[0,1])


