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
import scipy.signal as sig

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Spice.Netlist import Circuit

import lcapy as lcpy

import matplotlib.pyplot as plt
import numpy as np


w = sp.symbols('w', complex=False)
Ko = sp.symbols('Ko', complex=False)
eps_sq = sp.symbols('eps_sq', complex=False)
s = sp.symbols('s ', complex=True)

Tjw_sq = (1+ 0.25893) / (1+ 0.25893 * (8*w**4 - 8* w**2 + 1)**2 )

s11sq = sp.factor( 1 - Tjw_sq ).subs(w, s/sp.I)

s11sq = tc2.trim_func_s(s11sq)

s11 = tc2.modsq2mod_s(s11sq)

# no funca modsq2mod_s
s11 = (s*(s**2 + 0.5)) / (s**4 + 0.952805888465902*s**3 + 1.45391953054765*s**2 + 0.742613633524656*s + 0.275625642223404)

z1 = sp.simplify(sp.expand(sp.simplify(sp.expand(1+s11))/sp.simplify(sp.expand((1-s11)))))



# prototipo normalizado
num, den = sig.iirfilter(2, 1, rp=1, btype='lowpass', analog=True, ftype='cheby1')

# Resolución simbólica


# Ts = v2 / Vg/2 ojo con la definición!
# tol = 10**-2
# Ts = sp.nsimplify(num[0], tolerance = tol)/(s**4 + sp.nsimplify(den[1], tolerance = 10**-3)*s**3 + sp.nsimplify(den[2], tolerance = 10**-3)*s**2 + sp.nsimplify(den[3], tolerance = 10**-3)*s + sp.nsimplify(den[4], tolerance = tol) )
# Ts = num[0]*s**0/(s**4 + den[1]*s**3 + den[2]*s**2 + den[3]*s + den[4] )


s21sq_num = num * num 
s21sq_den = np.polymul(den, den * np.array([1,-1,1,-1,1]) )
 
s11sq_num = np.polysub(s21sq_den, s21sq_num )
s11sq_den =  s21sq_den

s11_num = tc2.modsq2mod(s11sq_num)
s11_den = tc2.modsq2mod(s11sq_den)

z1_num = np.polyadd(s11_num, s11_den)
z1_den = np.polysub(s11_den, s11_num)

tol = 10**-2
z1 = (sp.nsimplify(z1_num[0], tolerance = tol)*s**4 + sp.nsimplify(z1_num[1], tolerance = tol)*s**3 + sp.nsimplify(z1_num[2], tolerance = tol)*s**2 + sp.nsimplify(z1_num[3], tolerance = tol)*s + sp.nsimplify(z1_num[4], tolerance = tol) ) / (sp.nsimplify(z1_den[0], tolerance = tol)*s**4 + sp.nsimplify(z1_den[1], tolerance = tol)*s**3 + sp.nsimplify(z1_den[2], tolerance = tol)*s**2 + sp.nsimplify(z1_den[3], tolerance = tol)*s + sp.nsimplify(z1_den[4], tolerance = tol) ) 

# remoción parcial en infinito de z1
z3, l1 = tc2.remover_polo_infinito(z1)

l1 = l1 / s

# remoción parcial en infinito de y3
y5, c1 = tc2.remover_polo_infinito(1/z3)

c1 = c1 / s

# remoción parcial en infinito de z5
z7, l2 = tc2.remover_polo_infinito(1/y5)

l2 = l2 / s

y7, c2 = tc2.remover_polo_infinito(1/z7)

c2 = c2 / s

# Dibujo de la red sintetizada

# d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

# d, z1_lbl = tc2.dibujar_funcion_exc_abajo(d, 
#                                           'Z_{1}',  
#                                           z1.evalf(4), 
#                                           hacia_salida = True,
#                                           k_gap_width = 0.5)



pCircuit1 = Circuit('Filtro pasabajo Bessel 3er orden')

pCircuit1.SinusoidalVoltageSource('input', 'in', pCircuit1.gnd, amplitude=1)

pCircuit1.R('G', 'in',  1, 1)
pCircuit1.L('1',    1,  2, l1.evalf(4))
pCircuit1.C('1',    2,  pCircuit1.gnd, c1.evalf(4))
pCircuit1.L('2',    2,  'out', l2.evalf(4))
pCircuit1.R('L','out',  pCircuit1.gnd, 1)

# NGspice
simulation1 = pCircuit1.simulator(temperature=25, nominal_temperature=25)
# Xyce-serial
simulation1 = pCircuit1.simulator(simulator = 'xyce-serial', xyce_command='/home/mariano/XyceInstall/Serial/bin/Xyce', temperature=25, nominal_temperature=25)

analysis1 = simulation1.ac(start_frequency=0.001, stop_frequency=100, number_of_points=100,  variation='dec')

ff = 2*np.pi * np.array(analysis1.frequency.tolist())

figure, (axMod, axPha) = plt.subplots(2, sharex = True, figsize=(20, 10))

axMod.semilogx(ff, 20*np.log10(np.abs(analysis1['out'])))
axMod.semilogx(True)
axMod.grid(True)
axMod.grid(True, which='minor')
axMod.set_ylabel("Módulo [dB]")

plt.sca(axMod)
plt.title("Síntesis doblemente cargada: Bessel de 3er orden")

phase = np.unwrap(np.angle(analysis1['out']))
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




# Dibujamos y verificamos mediante LCAPY

cct = lcpy.Circuit("""
PG 1 0; down, v=V_{G}
RG 1 2 1; right
W  2 3; right
W  0 0_2; right
P1 3 0_2; down, v=V_{1}
W  0_2 0_3; right=2
L1 3 4 """ + str(l1.evalf(4)) + """; right
C1 4 0_3 """ + str(c1.evalf(4)) + """; down=2
W  0_3 0_4; right=2
L2 4 5 """ + str(l2.evalf(4)) + """; right
P2 5 0_4; down, v=V_{2}
W  0_4 0_5; right
W  5 6; right
RL 6 0_5 1; down=2
;draw_nodes=connections, label_nodes=False
;;\\node[black,draw,dashed,inner sep=7mm, fit= (L1) (C1) (L2) (0_3), label=Filter]{};""")

cct.draw()

#
print('Verificación de la transferencia via LCAPY')
tc2.print_latex('$ \\frac{V_2}{V_G/2}=' + sp.latex((cct.PG.transfer('P2')).evalf(4)) + '$')

