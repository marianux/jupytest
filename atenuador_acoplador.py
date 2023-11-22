#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicios para el uso de parámetros imagen donde se busca atenuar y acoplar 
dos niveles de impedancia.

@author: mariano
"""

import numpy as np
import splane as tc2

'''
    Enunciado:
    ----------
    
    1) Ej. 11 TP Filtrado Clásico

      Diseñe un atenuador de banda ancha que permita atenuar 30 dB intercalado en un cable coaxial de 75 ohms. 

    a) Obtenga una red que satisfaga los requerimientos.
    b) Verifique que el atenuador cumpla con la impedancia y atenuación prescrita. 

'''

r01 = 5 # Ω
r02 = 2 # Ω

atdb = 20 # dB

min_atdb = tc2.nepper2db( np.arccosh(np.sqrt(np.amax([ r02/r01,  r01/r02]) )))

if atdb < min_atdb:
    
    print('Atenuación mínima para el salto de impedancia =  {:3.3f} dB'.format(min_atdb))

    print('Asumiendo la atenuación mínima ...')
    # y un cachito más para ayudar numéricamente.
    atdb = min_atdb + 10**-3

# Calcular la matriz T a partir de los P. imagen
T1 = tc2.I2T(tc2.db2nepper(atdb), r01, r02)

# Convertir a Z
Z1 = tc2.T2Z(T1)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z1)

# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z1**-1)


TT = T1
TTi = T1**-1

print( 'Impedancia en el pto1 cargado con r02 =  {:3.3f}'.format( (TT[0,0] * r02 + TT[0,1])/(TT[1,0] * r02 + TT[1,1] ) ))
print( 'Impedancia en el pto2 cargado con r01  =  {:3.3f}'.format( (TTi[0,0] * r01 + TTi[0,1])/(TTi[1,0] * r01 + TTi[1,1] ) ))

TLZ = tc2.TabcdZ(r01) * T1 * tc2.TabcdZ(r02)
TLY = tc2.TabcdY(1/r01) * T1 * tc2.TabcdY(1/r02)

print( 'Atenuación de potencia como consigna =  {:3.3f} dB'.format(atdb))

print( '\n')
print( 'Verificación')
print( '------------')

print( 'Atenuación de tensión (cargado) =  {:3.3f} dB = {:3.3f} + {:3.3f}'.format( 20*np.log10(TLY[0,0]) + 10 * np.log10(r02/r01), 20*np.log10(TLY[0,0]), 10 * np.log10(r02/r01) ))
print( 'Atenuación de corriente (cargado) =  {:3.3f} dB = {:3.3f} + {:3.3f}'.format( 20*np.log10(TLZ[1,1]) + 10 * np.log10(r01/r02), 20*np.log10(TLZ[1,1]), 10 * np.log10(r01/r02) ))

