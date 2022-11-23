#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS14

@author: mariano
"""

import sympy as sp
import splane as tc2
from schemdraw import Drawing
from schemdraw.elements import  Resistor, Capacitor, Inductor


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
print('Punto 1')

zi1 = 75 # Ω
zi2 = 75 # Ω
atdb = 30 # dB

# Calcular la matriz T a partir de los P. imagen
T1 = tc2.I2T(atdb/8.686, zi1, zi2)

# Convertir a Z
Z1 = tc2.T2Z(T1)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z1)

# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z1**-1)

# Puerto 1
zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T1)
# Puerto 2
zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi1, T1**-1)


'''
    2) Ej. 12 TP Filtrado Clásico (Ampliado)

    Diseñe un acoplador/atenuador de banda ancha que permita interconectar un cable coaxil de 75 Ω a otro de 50 Ω. Es decir, la red en su entrada deberá permitir la conexión una ZI1=75Ω y en su salida ZI2=50Ω de impedancia característica.

    a) Obtenga una red que satisfaga los requerimientos y atenúe 5.72 dB en potencia.
    b) Verifique que el atenuador cumpla con la impedancia y atenuación prescrita. 
    c) ¿Podría dicha red acoplar sin atenuar? 
    d) Si la atenuación requerida fuera de 80 dB, ¿cómo cambiaría la red calculada en a)? Proponga una solución en caso que no se pueda implementar.
        
'''

# a)
print('Punto 2')
print('Inciso a)')

zi1 = 75 # Ω
zi2 = 50 # Ω
atdb = 5.72 # dB

# Calcular la matriz T a partir de los P. imagen
T2 = tc2.I2T(atdb/8.686, zi1, zi2)

# Convertir a Z
Z2 = tc2.T2Z(T2)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z2)

# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z2**-1)

# Puerto 1
zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T2)
# Puerto 2
zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi1, T2**-1)


# d)
print('Inciso d)')

zi1 = 75 # Ω
zi2 = 50 # Ω
atdb = 80 # dB

# Calcular la matriz T a partir de los P. imagen
# Matriz T total
T2dt = tc2.I2T(atdb/8.686, zi1, zi2)

# partimos en 4 secciones de 20 dB. La primera con salto de impedancia, 
# las otras 3 serán la misma
T2d1 = tc2.I2T(20/8.686, zi1, zi2)
T2d234 = tc2.I2T(20/8.686, zi2, zi2)

# Convertir a Z
Z2d1 = tc2.T2Z(T2d1)
Z2d234 = tc2.T2Z(T2d234)

# implementar como Tee
[Za, Zb, Zc] = tc2.dibujar_Tee(Z2d1)
# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z2d1**-1)

# Puerto 1
zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T2d1)
# Puerto 2
zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi1, T2d1**-1)

[Za, Zb, Zc] = tc2.dibujar_Tee(Z2d234)
# implementar como Pi
[Ya, Yb, Yc] = tc2.dibujar_Pi(Z2d234**-1)

# Puerto 1
zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T2d234)
# Puerto 2
zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T2d234**-1)

