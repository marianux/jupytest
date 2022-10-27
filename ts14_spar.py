#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS14 Análisis de redes mediante parámetros S

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor


'''
    Enunciado:
    ----------
    
    Calcular los parámetros S de las siguiente red:
    
    1------L=1---+---L=1-----2
                 |              
                C=2
                 |
    1'-----------+-----------2'
    
    
    a) ¿Qué tipo de comportamiento tiene la red analizada? Justifique utilizando alguno de los parámetros S.
    
    b) A partir del parámetro S11 y S21, explique el comportamiento de la red para:
    
        𝜔=0
    
      (centro de la banda de paso)
    𝜔=1
      (frecuencia de corte)
    𝜔→∞
    
          (centro de la banda de detención)
    
    Bonus:
    
        +10 🎓 Simulación circuital con LTspice. (Ver explicación de Agustín Alba Chicar 1h 48m)
        +10 🍺 Presentación en jupyter notebook

'''

s = sp.symbols('s ', complex=True)

#%% Verificación de la matriz de parámetros ABCD a partir de las matrices individuales
#   y su interconexión:
   
# Interconexion RG - L1 | C1 - L2 | RL. (Léase - interconexión serie, | derivación)
# el parámetro A = Vg/V2. Recordar que 
# S21 = V2/(Vg/2) * sqrt(Z02/Z01) (1 sin salto de impedancia)
Tabcd = sp.simplify(sp.expand(tc2.TabcdLZY_s( 1+s, 2*s) * tc2.TabcdLZY_s( s, 1)))

# también su dual cumple con la misma V2/(Vg/2)
Tabcd_dual = sp.simplify(sp.expand(tc2.TabcdLZY_s( 1, s) * tc2.TabcdLZY_s( 2*s, s) * tc2.TabcdY_s(1)))

# para validar los S desde los ABCD recurrimos a las equivalencias de parámetros
# luego de la interconexión L1 | C - L2
Spar_tee_desde_ABCD = tc2.Tabcd2S_s(sp.simplify(sp.expand(tc2.TabcdLZY_s( s, 2*s) * tc2.TabcdZ_s(s))))
Spar_pi_desde_ABCD = tc2.Tabcd2S_s(sp.simplify(sp.expand(tc2.TabcdLYZ_s( s, 2*s) * tc2.TabcdY_s(s))))


#%% Otra forma de verificación, mediante la interconexión de parámetros S

# Verificación de la matriz de parámetros S a partir de las matrices individuales
# y su interconexión:
    
# Inductor serie de 1 Hy. Z = S*1 Ω, Z0 = 1Ω
L1 = tc2.SparZ_s(s)
L2 = tc2.SparZ_s(s)

# Capa derivación de 2 F. Y = S*2 ℧, Z0 = 1℧
C1 = tc2.SparY_s(2*s)

# Convertimos a parámetros Ts e interconectamos en cascada
Tst_tee = sp.simplify(sp.expand(tc2.S2Ts_s(L1) * tc2.S2Ts_s(C1) * tc2.S2Ts_s(L2)))

# Volvemos a convertir a S
Spar_tee = tc2.Ts2S_s(Tst_tee)

# ahora verificamos los parámetros S de la red dual.
Tst_pi = sp.simplify(sp.expand( tc2.S2Ts_s(tc2.SparY_s(s)) * tc2.S2Ts_s(tc2.SparZ_s(2*s)) * tc2.S2Ts_s(tc2.SparY_s(s)) ))
Spar_pi = tc2.Ts2S_s(Tst_pi)

