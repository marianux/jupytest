#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:53:18 2021

@author: mariano
"""

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

zi1 = 50 # Ω
zi2 = 50 # Ω
atdb = 0 # dB
fase = 1 # radian

# Calcular la matriz T a partir de los P. imagen
T1 = tc2.I2T( ( atdb/8.686 + fase * 1j ), zi1, zi2)

# Convertir a Z
Z1 = tc2.T2Z(T1)

# implementar como Tee
[Za, Zb, Zc] = tc2.Z2tee(Z1)

# implementar como Pi
[Ya, Yb, Yc] = tc2.Y2Pi(Z1**-1)

# Puerto 1
zin1z(Za, Zb, Zc, Ya, Yb, Yc, zi2, T1)
# Puerto 2
zin2z(Za, Zb, Zc, Ya, Yb, Yc, zi1, T1**-1)

