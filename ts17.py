#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS16 ej 2

@author: mariano
"""

import sympy as sp
import splane as tc2

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Spice.Netlist import Circuit

import lcapy as lcpy

import matplotlib.pyplot as plt
import numpy as np

from schemdraw import Drawing
from schemdraw.elements import SourceSin, Resistor, Capacitor, Inductor


import os

os.environ['NGSPICE_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu/libngspice.so.0'
os.environ['NGSPICE'] = '/usr/bin/ngspice'

# Resolución simbólica

s = sp.symbols('s ', complex=True)
w = sp.symbols('w ', complex=False)

i = sp.I

n = 2

s21sq = sp.Rational('8/9') / (1 + w**(2*n))

s11sq = sp.factor( 1 - s21sq.subs(w, s/i) )

# Ts = v2 / Vg/2 ojo con la definición!
# Ts = sp.Rational('1/5')/(s**3 + sp.Rational('2')*s**2 + sp.Rational('2')*s + sp.Rational('1') )
# s11sq = sp.factor( 1- (Ts * Ts.subs(s, -s)) )

s11 = tc2.modsq2mod_s(s11sq)

z1 = (1+s11)/(1-s11)

# implemento como pasabajo escalera (Cauer1)
ko, imm_as_cauer, rem = tc2.cauer_LC(z1)

# Dibujo de la red sintetizada

d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads


d = tc2.dibujar_elemento_derivacion(d, SourceSin, 'Vg')

d = tc2.dibujar_elemento_serie(d, Resistor, "Rg=1")

d = tc2.dibujar_puerto_entrada(d,
                        voltage_lbl = ('+', '$V1$', '-'), 
                        current_lbl = '$I1$')

d, zz_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          z1, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)


d = tc2.dibujar_espacio_derivacion(d)


pCircuit1 = Circuit('Filtro pasabajo Butter {:d}º orden'.format(n))

# nodo de entrada
nodo = 1
pCircuit1.SinusoidalVoltageSource('input', nodo, pCircuit1.gnd, amplitude=1)

pCircuit1.R('gen', 'in',  1, 1)

# arranco en serie
bSerie = True
for koi in ko:

    this_val = (koi/s).evalf(4)
    
    if bSerie:        
        pCircuit1.L('L{:d}{:d}'.format(nodo, nodo+1), nodo,  nodo+1, this_val)
        
        d = tc2.dibujar_elemento_serie(d, Inductor, this_val)
        
    else:
        pCircuit1.C('C{:d}{:d}'.format(nodo+1,0), nodo+1,  pCircuit1.gnd, this_val )
        
        d = tc2.dibujar_elemento_derivacion(d, Capacitor, this_val)
        
        nodo += 1
    
    # voy armando la L  z serie - y derivacion
    bSerie = not bSerie
    
      
if not bSerie:        
    # ultimo elemento en serie, rem en siemens
    nodo += 1
    rem = 1/rem
    
# nodo es el puerto de salida    
pCircuit1.R('load', nodo,  pCircuit1.gnd, rem.evalf(4))

d = tc2.dibujar_puerto_salida(d,
                        voltage_lbl = ('+', '$V2$', '-'), 
                        current_lbl = '$I2$')

d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_elemento_derivacion(d, Resistor, rem)

display(d)


# NGspice
simulation1 = pCircuit1.simulator(temperature=25, nominal_temperature=25)
# Xyce-serial
# simulation1 = pCircuit1.simulator(simulator = 'xyce-serial', xyce_command='/home/mariano/XyceInstall/Serial/bin/Xyce', temperature=25, nominal_temperature=25)

analysis1 = simulation1.ac(start_frequency=0.001, stop_frequency=100, number_of_points=100,  variation='dec')

ff = 2*np.pi * np.array(analysis1.frequency.tolist())

figure, (axMod, axPha) = plt.subplots(2, sharex = True, figsize=(20, 10))

axMod.semilogx(ff, 20*np.log10(np.abs(analysis1['{:d}'.format(nodo)])))
axMod.semilogx(True)
axMod.grid(True)
axMod.grid(True, which='minor')
axMod.set_ylabel("Módulo [dB]")

plt.sca(axMod)
plt.title("Síntesis doblemente cargada: Bessel de 3er orden")

phase = np.unwrap(np.angle(analysis1['{:d}'.format(nodo)]))
delay = - np.diff(phase) / np.diff(ff)
delay = np.concatenate(([delay[0]], delay))
axPha.semilogx(ff, delay)
axPha.grid(True)
axPha.grid(True, which='minor')
axPha.set_xlabel("Frecuencia [rad/s]")
axPha.set_ylabel("Retardo [s]")
# axe.set_yticks # Fixme:
# plt.yticks((-math.pi, -math.pi/2,0, math.pi/2, math.pi),
#               (r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"))




# # Dibujamos y verificamos mediante LCAPY

# cct = lcpy.Circuit("""
# PG 1 0; down, v=V_{G}
# RG 1 2 1; right
# W  2 3; right
# W  0 0_2; right
# P1 3 0_2; down, v=V_{1}
# W  0_2 0_3; right=2
# L1 3 4 """ + str(l1.evalf(4)) + """; right
# C1 4 0_3 """ + str(c1.evalf(4)) + """; down=2
# W  0_3 0_4; right=2
# L2 4 5 """ + str(l2.evalf(4)) + """; right
# P2 5 0_4; down, v=V_{2}
# W  0_4 0_5; right
# W  5 6; right
# RL 6 0_5 1; down=2
# ;draw_nodes=connections, label_nodes=False
# ;;\\node[black,draw,dashed,inner sep=7mm, fit= (L1) (C1) (L2) (0_3), label=Filter]{};""")

# cct.draw()

# #
# print('Verificación de la transferencia via LCAPY')
# tc2.print_latex('$ \\frac{V_2}{V_G/2}=' + sp.latex((cct.PG.transfer('P2')).evalf(4)) + '$')

