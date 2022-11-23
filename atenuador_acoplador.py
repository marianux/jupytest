#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicios para el uso de parámetros imagen donde se busca atenuar y acoplar 
dos niveles de impedancia.

@author: mariano
"""

import numpy as np
import splane as tc2


def zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, TT):
    
    # a la TC1
    print( 'Impedancia en el pto1 Tee ´a la TC1´ [(Zc+Zi2)//Zb]+Za =  {:3.3f}'.format(tc2.pp( Zc + zi2, Zb) + Za) )
    print( 'Impedancia en el pto1 Pi ´a la TC1´ [(1/Zi2+Yc)//Yb]+Ya =  {:3.3f}'.format( 1/(tc2.pp( (1/zi2 + Yc), Yb) + Ya)) )
    # a la TC2
    print( 'Impedancia en el pto1 ´a la TC2´ [(A*Zi2+B)/(C*Zi2+D) =  {:3.3f}'.format( (TT[0,0] * zi2 + TT[0,1])/(TT[1,0] * zi2 + TT[1,1] ) ))
    
    return()

def zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi1, TTi):
    
    # a la TC1
    print( 'Impedancia en el pto2 Tee ´a la TC1´ [(Za+Zi1)//Zb]+Zc =  {:3.3f}'.format(tc2.pp( Za + zi1, Zb) + Zc) )
    print( 'Impedancia en el pto2 Pi ´a la TC1´ [(1/Zi1+Ya)//Yb]+Yc =  {:3.3f}'.format( 1/(tc2.pp( (1/zi1 + Ya), Yb) + Yc)) )
    # a la TC2
    print( 'Impedancia en el pto2 ´a la TC2´ [(AA*Zi1+BB)/(CC*Zi1+DD) =  {:3.3f}'.format( (TTi[0,0] * zi1 + TTi[0,1])/(TTi[1,0] * zi1 + TTi[1,1] ) ))
    print( 'T_inversa = [AA, BB],[CC, DD] = T**-1 ')
    
    return()

'''
    Enunciado:
    ----------
    
    1) Ej. 11 TP Filtrado Clásico

      Diseñe un atenuador de banda ancha que permita atenuar 30 dB intercalado en un cable coaxial de 75 ohms. 

    a) Obtenga una red que satisfaga los requerimientos.
    b) Verifique que el atenuador cumpla con la impedancia y atenuación prescrita. 

'''

r01 = 1 # Ω
r02 = np.sqrt(5)/5 # Ω

atdb = 8.324 # dB


# Calcular la matriz T a partir de los P. imagen
T1 = tc2.I2T(tc2.db2nepper(atdb), r01, r02)

# Convertir a Z
Z1 = tc2.T2Z(T1)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z1)

# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z1**-1)


TT = T1
TTi = T1**-1 fermmmmmin

print( 'Impedancia en el pto1 cargado con r02 =  {:3.3f}'.format( (TT[0,0] * r02 + TT[0,1])/(TT[1,0] * r02 + TT[1,1] ) ))
print( 'Impedancia en el pto2 cargado con r01  =  {:3.3f}'.format( (TTi[0,0] * r01 + TTi[0,1])/(TTi[1,0] * r01 + TTi[1,1] ) ))

TLZ = tc2.TabcdZ(r01) * T1 * T1 * tc2.TabcdZ(r02)
TLY = tc2.TabcdY(1/r01) * T1 * tc2.TabcdY(1/r02)

print( 'Atenuación de potencia como consigna =  {:3.3f} dB'.format(atdb))
print( 'Atenuación mínima para el salto de impedancia =  {:3.3f} dB'.format(tc2.nepper2db( np.arccosh(np.sqrt(np.amax([ r02/r01,  r01/r02]) )))) ) 

print( '\n')
print( 'Verificación')
print( '------------')

print( 'Atenuación de tensión (cargado) =  {:3.3f} dB = {:3.3f} + {:3.3f}'.format( 20*np.log10(TLY[0,0]) + 10 * np.log10(r02/r01), 20*np.log10(TLY[0,0]), 10 * np.log10(r02/r01) ))
print( 'Atenuación de corriente (cargado) =  {:3.3f} dB = {:3.3f} + {:3.3f}'.format( 20*np.log10(TLZ[1,1]) + 10 * np.log10(r01/r02), 20*np.log10(TLZ[1,1]), 10 * np.log10(r01/r02) ))






