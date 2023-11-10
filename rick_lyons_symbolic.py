#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5: Filtros digitales

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
from splane import pzmap, bodePlot
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


#%% Resolución simbólica

z = sp.symbols('z', complex=True)
D, U = sp.symbols('D U', real=True, positive=True)

# moving average
Tma = 1/D * (1-z**(-D))/(1-z**(-1))

# delay line of (D-1)/2
Tdl =  z**(-(D-1)/2)

num, den = (Tdl - Tma).as_numer_denom()

num = (sp.expand(num/(D*z**(D+1)))).powsimp()
den = (sp.expand(den/(D*z**(D+1)))).powsimp()

Tdc_removal = num/den

display(Tdc_removal)

# Según Rick Lyons, este sistema sería muy bueno para implementarse
# con D múltiplo de 2**N, dado que el escalado por D sería simplemente 
# una rotación a nivel de bits de N veces a la derecha, y su implementación
# no necesitaría de multiplicaciones. Sin embargo esta elección impone un 
# retardo no entero. Por esta razón se opta por poner dos (incluso cuatro) 
# sistemas idénticos en cascada.

# Probamos primero con 2 moving average

Tdc_removal_2 = z**-(D-1) - Tma**2

# emprolijamos la expresion
num, den = Tdc_removal_2.as_numer_denom()
num = sp.simplify(sp.expand(num).powsimp())
den = sp.simplify(sp.expand(den).powsimp())

num = (sp.expand(num/(D**2*z**(2*D+2)))).powsimp()

def recursive_exploration( this_expr, poly_var, sorted_expr ):
    
    for aux_terms in this_expr.as_poly(poly_var).all_terms():
        
        if len(sp.expand(aux_terms[1]).as_poly(poly_var).all_terms()) == 1:
        
            sorted_expr = sorted_expr + aux_terms[1] * poly_var**aux_terms[0][0]
        
        else:
        
            sorted_expr = recursive_exploration( aux_terms[1] * poly_var**aux_terms[0][0], poly_var, sorted_expr )

    return sorted_expr

sorted_expr = 0

poly_var = z**-D

sorted_expr = recursive_exploration( num, poly_var, sorted_expr )

den = (sp.expand(den/(D**2*z**(2*D+2)))).powsimp()

Tdc_removal_2 = num/den

display(Tdc_removal_2)

# Ahora con 4 moving average

# Tdc_removal_4 = z**-(2*D-2) - Tma**4

# # emprolijamos la expresion
# num, den = Tdc_removal_4.as_numer_denom()
# num = (sp.expand(num/(D**4*z**(4*D+4)))).powsimp()
# den = (sp.expand(den/(D**4*z**(4*D+4)))).powsimp()

# Tdc_removal_4 = num/den

# display(Tdc_removal_4)

# D_par = 16

# num_rl4 = sp.zeros(4*D_par,1)

# num*z
# sp.expand(num*z).powsimp()

# for a_term in num.as_ordered_terms():
    
#     z_coeff, var = a_term.as_coeff_Mul()
    
#     zz, z_exp = var.as_base_exp()
    
#     num_rl4[] = z_coeff.subs(D, D_par)
    


#%% Parte numérica 

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt


fs = 1000 # Hz (Normalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2


# fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 

## Rick Lyons ECG filter
dd = 16
uu = 4
# num_rl = np.hstack([-1/dd**2, np.zeros(uu*(dd-1)-1), 1, np.zeros(uu-1), (2/dd**2-2) ])
num_rl = np.hstack([-1/dd**2, np.zeros(uu*dd-5), 1, np.zeros(uu-1), (2/dd**2-2) ])
num_rl = np.hstack([ num_rl, np.flip(num_rl[:-1]) ])
den_rl = np.hstack([1, np.zeros(uu-1), -2, np.zeros(uu-1), 1])

# num_rl = np.hstack([-1/dd**2, np.zeros(uu*(dd-1)-1), 1, np.zeros(uu-1), (2/dd**2-2), np.zeros(uu-1), 1, np.zeros(uu*(dd-1)-1), -1/dd**2])
# den_rl = np.hstack([1, np.zeros(uu-1), -2, np.zeros(uu-1), 1])

demora_rl = int(uu*(dd-1))

den = 1.0

N = 4000
w_rad  = np.append(np.logspace(-2, 0.8, N//4), np.logspace(0.9, 1.6, N//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, N//2, endpoint=True) ) / nyq_frec * np.pi

_, hh_rl = sig.freqz(num_rl, den_rl, w_rad)

w = w_rad / np.pi * nyq_frec

plt.close('all')

plt.figure(1)
plt.clf()

plt.plot(w, 20 * np.log10(abs(hh_rl)), label='FIR-Rick {:d}'.format(num_rl.shape[0]))
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.axis([0, 500, -60, 5 ]);

plt.grid()


#%%

# allQ = np.array([0.5, np.sqrt(2)/2, 5])
# allfs = np.array([ 1, 2, 4])

# this_q = np.sqrt(2)/2 # Butter
# this_fs = allfs[0] # fs

# plt.close('all')

# for this_fs in allfs:
    
#     for this_q in allQ[:-1]:

    
#         k = 2 * this_fs
            
#         kz2 = this_q * k**2  + this_q + k
#         kz1 = -2 * this_q * k**2 + 2 * this_q
#         kz0 = this_q * k**2  + this_q - k
        
#         numz =  this_q * np.array([1, 2, 1])
#         denz =  np.array([kz2, kz1, kz0])
        
#         my_df = sig.TransferFunction(numz, denz, dt=1/fs)
        
#         #filter_description='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs)
#         pzmap(my_df, annotations = False,  fig_id=1)
        
#         bodePlot(my_df, fig_id=2, digital = True, fs = this_fs)
        
    
#     # el último le ponemos anotación para que quede lindo el gráfico
#     this_q = allQ[-1]
#     k = 2 * this_fs
        
#     kz2 = this_q * k**2  + this_q + k
#     kz1 = -2 * this_q * k**2 + 2 * this_q
#     kz0 = this_q * k**2  + this_q - k
    
#     numz =  this_q * np.array([1, 2, 1])
#     denz =  np.array([kz2, kz1, kz0])
    
#     my_df = sig.TransferFunction(numz, denz, dt=1/fs)
    
#     pzmap(my_df, annotations = False, filter_description='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs), fig_id=1)

#     bodePlot(my_df, fig_id=2, digital = True, fs = this_fs, filter_description ='Q={:3.3f} - fs={:3.3f}'.format(this_q, this_fs))

#     # bodePlot(my_df, fig_id=2, digital = True)
#     # bodePlot(myFilter, fig_id='none', axes_hdl='none', filter_description = '', npoints = 1000, digital = False, fs = 2*np.pi ):
   


    
    