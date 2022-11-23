#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicios para el uso de parámetros imagen donde se busca atenuar y acoplar 
dos niveles de impedancia.

@author: mariano
"""

import numpy as np
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import SourceSin, Resistor

'''
    Enunciado:
    ----------
    
    1) Diseñe un atenuador de banda ancha ...

    a) Obtenga una red que satisfaga los requerimientos.
    b) Verifique que el atenuador cumpla con la impedancia y atenuación prescrita. 

'''

r01 = 1 # Ω
# si cambia el salto de impedancia hacia mayor nivel, cambia la topología
# y el cálculo de la atenuación mínima.
r02 = 1/5 # Ω

# plantearemos varios saltos de impedancia.
cant_saltos = 4

# tipo_espaciado = 'lineal'
# saltos = np.linspace( r01, r02, num = (cant_saltos+1))
# tipo_espaciado = 'log'
saltos = np.logspace( np.log10(r01), np.log10(r02), num = (cant_saltos+1))


At_cascada = 0
TTcascada = np.eye(2)

d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

d = tc2.dibujar_elemento_derivacion(d, SourceSin, 'Vg')

d = tc2.dibujar_elemento_serie(d, Resistor, "r01={:3.3f}".format(r01))

d = tc2.dibujar_puerto_entrada(d,
                               port_name = 'In' )

for ii in range(cant_saltos):
    
    if r02 < r01:
        min_at_nepper = np.arccosh(np.sqrt( saltos[ii]/saltos[ii+1] ))
    else:
        min_at_nepper = np.arccosh(np.sqrt( saltos[ii+1]/saltos[ii] ))
    
    At_cascada += min_at_nepper
    
    TTi = tc2.I2T(min_at_nepper, saltos[ii], saltos[ii+1])
    
    # Convertir a Z
    Zi = tc2.T2Z(TTi)
    
    # implementar como Tee, como forzamos la at mínima, 
    # Rci tiene que ser despreciable
    if r02 < r01:
        Rai = Zi[0,0] - Zi[0,1] 
        
    Rbi = Zi[0,1] 

    if r02 > r01:
        Rci = Zi[1,1] - Zi[0,1] 
    
    # impedancia vista desde la sección i-esima
    Rii = (TTi[0,0] * saltos[ii+1] + TTi[0,1])/(TTi[1,0] * saltos[ii+1] + TTi[1,1] )
    
    d,_ = tc2.dibujar_funcion_exc_abajo(d, 
                                              'R',  
                                              Rii, 
                                              hacia_salida = True,
                                              k_gap_width = 0.5)
    
    d,_ = tc2.dibujar_funcion_exc_arriba(d, 
                                              'At',  
                                              tc2.nepper2db(min_at_nepper), 
                                              hacia_salida = True,
                                              k_gap_width = 0.5)
    
    if r02 < r01:
        d = tc2.dibujar_elemento_serie(d, Resistor, Rai )
    
    d = tc2.dibujar_elemento_derivacion(d, Resistor, Rbi )

    if r02 > r01:
        d = tc2.dibujar_elemento_serie(d, Resistor, Rci )

    TTcascada = np.matmul(TTcascada, TTi)
    
    print( 'Impedancia en el pto1 cargado con r02i = {:3.3f}; ro1i =  {:3.3f}'.format(saltos[ii+1], Rii))
    

d = tc2.dibujar_puerto_salida(d)

d = tc2.dibujar_espacio_derivacion(d)
d = tc2.dibujar_espacio_derivacion(d)

d = tc2.dibujar_elemento_derivacion(d, Resistor, "r02={:3.3f}".format(r02) )

display(d)        

TT = TTcascada
TTi = TTcascada**-1

# Convertir a Z
Z1 = tc2.T2Z(TT)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z1)

print( 'Impedancia en el pto1 cargado con r02 =  {:3.3f}'.format( (TT[0,0] * r02 + TT[0,1])/(TT[1,0] * r02 + TT[1,1] ) ))
print( 'Impedancia en el pto2 cargado con r01  =  {:3.3f}'.format( (TTi[0,0] * r01 + TTi[0,1])/(TTi[1,0] * r01 + TTi[1,1] ) ))

TLZ = tc2.TabcdZ(r01) * TT * tc2.TabcdZ(r02)
TLY = tc2.TabcdY(1/r01) * TT * tc2.TabcdY(1/r02)

if r02 < r01:
    min_at_consigna = tc2.nepper2db(np.arccosh(np.sqrt( r01/r02 )))
else:
    min_at_consigna = tc2.nepper2db(np.arccosh(np.sqrt( r02/r01 )))
    
print( 'Atenuación de potencia como consigna =  {:3.3f} dB'.format(min_at_consigna))

At_cascada = tc2.nepper2db(At_cascada)
print( 'Atenuación de potencia cascada =  {:3.3f} dB'.format(At_cascada))

