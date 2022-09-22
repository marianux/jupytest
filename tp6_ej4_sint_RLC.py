#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP6 ej 4: Sintetizar una red RLC.

            +---L1----+
    o---R1--o---R2----o--+
                      |  |      
                      C1 R3
                      |  |
    o-----------------o--+

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Line


# Resolución simbólica

s = sp.symbols('s ', complex=True)

# Sea la siguiente función de excitación
ZZ = (s**2+10*s+24)/(s**2+12*s+20)

# Halle los valores de los componentes de la topología resultante.
# tener en cuenta que no se puede remover los valores reales
# ya que en ninguno caso los residuos en inf o dc se corresponden
# con algún resistor solo
# Z(0) = R1+R3
# Z(oo) = R1+R2

# Consignas del ejercicio: resonancias de los tanques RL-RC
sigma1 = 2
sigma2 = 10

# La topología circuital guía las remociones, en este caso empezamos
# por sigma1. Ver qu
Z2, Z1, R3, C1 = tc2.remover_polo_sigma(ZZ, sigma1, isImpedance = True, isRC = True )

# Remuevo el tanque RL y lo que resta sería R1.
R1, Z3, R2, L1 = tc2.remover_polo_sigma(Z2, sigma2, isImpedance = True, isRC = False )


# Dibujamos la red resultante:
    
d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d = tc2.dibujar_puerto_entrada(d,
                        voltage_lbl = ('+', '$V$', '-'), 
                        current_lbl = '$I$')

d, zz_lbl = tc2.dibujar_funcion_exc_abajo(d, 
                                          'Z',  
                                          ZZ, 
                                          hacia_salida = True,
                                          k_gap_width = 0.5)

d = tc2.dibujar_elemento_serie(d, Resistor, R1)

d = tc2.dibujar_tanque_RL_serie(d, R2, L1)
    
d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_tanque_RC_serie(d, R3, C1)

# cerramos el dibujo
d += Line().right().length(d.unit*.25)
d += Line().down()
d += Line().left().length(d.unit*.25)

display(d)

