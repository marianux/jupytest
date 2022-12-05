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
from splane import s, w
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

import numpy as np

from schemdraw import Drawing
from schemdraw.elements import SourceSin, Resistor, Capacitor, Inductor

# Tipo de aproximación
# aprox = 'butter'
aprox = 'cheby'

# Salto de impedancia:
# Para órdenes impares el salto de impedancia es libre
# En el caso de los pares, el único que no es posible es que R01 == R02.
R01 = 1
R02 = 1
# orden del filtro
nn = 3
# ripple
alfa_max = 3 # dB

bImpar = True
if nn % 2 == 0:
    # nn es par
    bImpar = False
    
    if aprox == 'cheby':
        assert R01 != R02, 'Si nn es par NO es posible R01 == R02. '


# El signo de S11 determina si R02 será mayor o menor a 1 (R01)
# Recordar que el signo de S11 es arbitrario dada una S21
if R02 >= 1:
    s11_sign = 1
else:
    s11_sign = -1
    
# A veces puede tener sentido forzar el signo para hallar la red
# dual.
# s11_sign = -1

# A veces puede tener sentido forzar el eps_sq (ej Butter )
# eps_sq = 10**(alfa_max/10) - 1
eps_sq = 1

if aprox == 'cheby':

    if bImpar:
    
        Ko = 1 - ((R02-1)/(R02+1))**2
        
    else:
        
        Gmin = 1 - ((R02-1)/(R02+1))**2
        # Ko debe ser menor a 1 para redes pasivas.
        Ko  = (1+eps_sq) * Gmin

else:

    Ko = 1 - ((R02-1)/(R02+1))**2
    
if Ko > 1:

    
    if bImpar:
        
        Ko = 1 - 1/100
        # si lo fuera ajustamos el ripple para cumplir con Gmin.
        eps_sq_recalc = (Ko - Gmin) / Gmin
        
        alfa_max_recalc = 10 * np.log10(eps_sq_recalc + 1)
        
        alfa_max = alfa_max_recalc
        
        eps_sq = eps_sq_recalc
        
        print( 'Se ajustó el ripple a {:3.3f} dB para cumplir con el requerimiento de R02 = {:3.3f} Ω'.format(alfa_max, R02) )

        # Verificación de R02. Siempre verificar que el salto de impedancia es
        # el impuesto por R02 y el signo de S11
        R02_recalc = ((1+np.sqrt(1-Gmin))/(1-np.sqrt(1-Gmin)))**(s11_sign)
        
    else:
        
        Ko = 1

        R02_recalc = ((1+np.sqrt(1-Ko))/(1-np.sqrt(1-Ko)))**(s11_sign)

# Para orden par Ko = 1 + eps**2
# recordar que eps**2 = 10**(alfa_max/10) - 1
# alfa_max dB

if aprox == 'cheby':
   
    Tjw_sq = (Ko) / (1+ (eps_sq) * (tc2.Chebyshev_polynomials(nn))**2 )
    
else:
        
    Tjw_sq = (Ko) / (1+ (eps_sq) * w**(2*nn))

s11sq = sp.simplify(sp.expand(sp.factor( 1 - Tjw_sq ).subs(w, s/sp.I)))

# s11sq = sp.simplify(sp.expand(s11sq.subs(Ko, 10**(1/10))))

# s11sq = tc2.trim_func_s(s11sq)


# s11_num = sp.simplify(sp.expand(sp.factor(( ((10**(1/10)-1) * (8*w**4 - 8* w**2 + 1)).subs(w, s/sp.I) ))))
# s11_den = tc2.modsq2mod_s(sp.simplify(sp.expand(sp.factor((1 + (10**(1/10)-1) * (8*w**4 - 8* w**2 + 1)**2 )))).subs(w, s/sp.I))
# s11 = s11_num / s11_den
s11 = s11_sign * tc2.modsq2mod_s(s11sq)

num, den = s11.as_numer_denom()
z1 = sp.simplify(sp.expand(den + num) / sp.expand(den - num))
# z1 = sp.simplify(sp.expand(sp.simplify(sp.expand(1+s11))/sp.simplify(sp.expand((1-s11)))))
# z1 = sp.simplify(sp.expand(sp.simplify(sp.expand(1+s11.evalf(10)))/sp.simplify(sp.expand((1-s11.evalf(10))))))

_, koo = tc2.remover_polo_infinito(z1)

if koo.is_zero:
    bImpedancia = False
    immitance = 1/z1
else:
    immitance = z1
    bImpedancia = True
    
koo, imm_as_cauer, remainder = tc2.cauer_LC(immitance, remover_en_inf = True)


# Dibujo de la red sintetizada Imagen + LTspice

circ_name = 'ecualizador_{:s}_orden_{:d}_ripple_{:d}dB_r01_{:d}_r02_{:d}'.format(aprox, nn, alfa_max, R01, R02)

circ_hdl = tc2.ltsp_nuevo_circuito(circ_name)

tc2.ltsp_etiquetar_nodo(circ_hdl, node_label='vi')

d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads


d = tc2.dibujar_elemento_derivacion(d, SourceSin, 'Vg')

d = tc2.dibujar_elemento_serie(d, Resistor, "Rg=1Ω")

d = tc2.dibujar_puerto_entrada(d,
                        voltage_lbl = ('+', '$V1$', '-'), 
                        current_lbl = '$I1$')

d, zz_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          z1.evalf(5), 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_espacio_derivacion(d)

if bImpedancia:
    bEnSerie = True
else:
    bEnSerie = False

for this_inf_pole in koo:

    if bEnSerie:

        ind_value = (this_inf_pole/s).evalf(5)
        
        d = tc2.dibujar_elemento_serie(d, Inductor, ind_value)
        
        tc2.ltsp_ind_serie(circ_hdl, ind_value) 
        
    else:

        cap_value = (this_inf_pole/s).evalf(5)
        
        d = tc2.dibujar_elemento_derivacion(d, Capacitor, cap_value)
        
        tc2.ltsp_capa_derivacion(circ_hdl, cap_value) 
        
    # forma de escalera serie/deriación
    bEnSerie = not bEnSerie
    

d = tc2.dibujar_puerto_salida(d,
                        voltage_lbl = ('+', '$V2$', '-'), 
                        current_lbl = '$I2$')

d = tc2.dibujar_espacio_derivacion(d)
d = tc2.dibujar_espacio_derivacion(d)
d = tc2.dibujar_espacio_derivacion(d)

if bEnSerie:
    # último elemento en derivación, resto de la división en admitancia
    d = tc2.dibujar_elemento_derivacion(d, Resistor, (1/remainder).evalf(4) )
else:
    d = tc2.dibujar_elemento_derivacion(d, Resistor, remainder.evalf(4) )
        
display(d)

tc2.ltsp_etiquetar_nodo(circ_hdl, node_label='vo')

circ_hdl.writelines('TEXT -48 304 Left 2 !.param RG={:d} RL={:d}'.format(R01, R02))

circ_hdl.close()

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

