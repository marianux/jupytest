#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano
"""

alfa_max = 1
ws = 10

# cuentas auxiliares

ee = 10**(alfa_max/10)-1

for nn in range(2,5):
    
    alfa_min_b = 10*np.log10(1 + ee * ws**(2*nn))
    alfa_min_c = 10*np.log10(1 + ee * np.cosh(nn * np.arccosh(ws))**2 )
    
    print( 'nn {:d} - alfa_min_butter {:f} - alfa_min_cheby {:f}'.format(nn, alfa_min_b, alfa_min_c) )



# verificación MP
z,p,k = sig.buttap(nn)

num, den = sig.zpk2tf(z,p,k)
num, den = sig.lp2lp(num, den, ee**(-1/2/nn))

# verificación Cheby

np.roots([4*np.sqrt(ee), aa[2], np.sqrt(2*aa[2]+9*ee), 1 ])

z,p,k = sig.cheb1ap(nn, alfa_max)