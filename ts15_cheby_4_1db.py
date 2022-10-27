#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS15 Consigna:

    Diseñe el cuadripolo A para que se comporte como:

    filtro pasa bajos Chebyshev de 4to. orden, 1dB de ripple. (recordatorio en el siguiente video)

(Recordatorio de Cheby - Enlace 👇)
https://www.youtube.com/watch?v=lYirQkTkq-w&list=PLlD2eDv5CIe-0IZ3VOP0aQPTgAn9NMoKY&index=2

    no disipativo
    normalizado en frecuencia e impedancia


    respetando la siguiente topología (Cuadripolo doblemente cargado)


    1. Obtenga la impedancia de entrada al cuadripolo A, cargado con un resistor de 1Ω a la salida.
    2. Sintetice A como un cuadripolo escalera.
    3. Simule el comportamiento de la red en LTspice graficando S21 y S11 en función de la frecuencia. (Ver explicación de Agustín Alba Chicar 1h 48m)
    4. Explique el comportamiento de A a partir de los valores de S11 en las siguientes frecuencias:
        a. centro de la banda de paso
        b. frecuencia de corte
        c. transición y centro de la banda de detenida
    5. Modifique el circuito para que la frecuencia de corte sea 2 π 10⁶ rad/s y la resistencia del generador sea 50Ω.


@author: mariano
"""

import sympy as sp
import splane as tc2

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

import numpy as np

from schemdraw import Drawing
from schemdraw.elements import SourceSin, Resistor, Capacitor, Inductor


w = sp.symbols('w', real=True)
Ko = sp.symbols('Ko', real=True)
eps_sq = sp.symbols('eps_sq', real=True)
s = sp.symbols('s ', complex=True)

# Para orden par Ko = 1 + eps**2
# recordar que eps**2 = 10**(alfa_max/10) - 1
# alfa_max 1 dB
Tjw_sq = (9/10) / (1+ (10**(1/10)-1) * (8*w**4 - 8* w**2 + 1)**2 )

s11sq = sp.simplify(sp.expand(sp.factor( 1 - Tjw_sq ).subs(w, s/sp.I)))

# s11sq = sp.simplify(sp.expand(s11sq.subs(Ko, 10**(1/10))))

# s11sq = tc2.trim_func_s(s11sq)

s11 = tc2.modsq2mod_s(s11sq)

# no funca modsq2mod_s

z1 = sp.simplify(sp.expand(sp.simplify(sp.expand(1+s11))/sp.simplify(sp.expand((1-s11)))))


# remoción parcial en infinito de z1
z3, l1 = tc2.remover_polo_infinito(z1)

l1 = l1 / s

z3 = tc2.trim_func_s(sp.simplify(sp.expand(z3)))

# remoción parcial en infinito de y3
y5, c1 = tc2.remover_polo_infinito(1/z3)

c1 = c1 / s

y5 = tc2.trim_func_s(sp.simplify(sp.expand(y5)))

# remoción parcial en infinito de z5
z7, l2 = tc2.remover_polo_infinito(1/y5)

l2 = l2 / s

z7 = tc2.trim_func_s(sp.simplify(sp.expand(z7)))

y7, c2 = tc2.remover_polo_infinito(1/z7)

c2 = c2 / s

# Dibujo de la red sintetizada

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


d = tc2.dibujar_elemento_serie(d, Inductor, l1.evalf(4))

d = tc2.dibujar_elemento_derivacion(d, Capacitor, c1.evalf(4))

d = tc2.dibujar_elemento_serie(d, Inductor, l2.evalf(4))

d = tc2.dibujar_elemento_derivacion(d, Capacitor, c2.evalf(4))

d = tc2.dibujar_puerto_salida(d,
                        voltage_lbl = ('+', '$V2$', '-'), 
                        current_lbl = '$I2$')

d = tc2.dibujar_espacio_derivacion(d)
d = tc2.dibujar_espacio_derivacion(d)
d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_elemento_derivacion(d, Resistor, 1/y7.evalf(4) )

display(d)



# pCircuit1 = Circuit('Filtro pasabajo Bessel 3er orden')

# pCircuit1.SinusoidalVoltageSource('input', 'in', pCircuit1.gnd, amplitude=1)

# pCircuit1.R('G', 'in',  1, 1)
# pCircuit1.L('1',    1,  2, l1.evalf(4))
# pCircuit1.C('1',    2,  pCircuit1.gnd, c1.evalf(4))
# pCircuit1.L('2',    2,  'out', l2.evalf(4))
# pCircuit1.R('L','out',  pCircuit1.gnd, 1)

# # NGspice
# simulation1 = pCircuit1.simulator(temperature=25, nominal_temperature=25)
# # Xyce-serial
# simulation1 = pCircuit1.simulator(simulator = 'xyce-serial', xyce_command='/home/mariano/XyceInstall/Serial/bin/Xyce', temperature=25, nominal_temperature=25)

# analysis1 = simulation1.ac(start_frequency=0.001, stop_frequency=100, number_of_points=100,  variation='dec')

# ff = 2*np.pi * np.array(analysis1.frequency.tolist())

# figure, (axMod, axPha) = plt.subplots(2, sharex = True, figsize=(20, 10))

# axMod.semilogx(ff, 20*np.log10(np.abs(analysis1['out'])))
# axMod.semilogx(True)
# axMod.grid(True)
# axMod.grid(True, which='minor')
# axMod.set_ylabel("Módulo [dB]")

# plt.sca(axMod)
# plt.title("Síntesis doblemente cargada: Bessel de 3er orden")

# phase = np.unwrap(np.angle(analysis1['out']))
# delay = - np.diff(phase) / np.diff(ff)
# delay = np.concatenate(([delay[0]], delay))
# axPha.semilogx(ff, delay)
# axPha.grid(True)
# axPha.grid(True, which='minor')
# axPha.set_xlabel("Frecuencia [rad/s]")
# axPha.set_ylabel("Retardo [s]")
# # axe.set_yticks # Fixme:
# # plt.yticks((-math.pi, -math.pi/2,0, math.pi/2, math.pi),
# #               (r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"))




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

