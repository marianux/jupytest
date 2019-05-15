#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import analyze_sys, pzmap, grpDelay, bodePlot, pretty_print_lti
    

nn = 2 # orden
ripple = 1 # dB

eps = np.sqrt(10**(ripple/10)-1)

# Diseño un Butter.
z,p,k = sig.buttap(nn)
num_lp, den_lp = sig.zpk2tf(z,p,k)

# paso de un Butter. a maxima planicidad si eps != 1
num_lp_butter, den_lp_butter = sig.lp2lp(num_lp,den_lp, eps**(-1/nn) )

# obtengo la transferencia normalizada del pasabanda
num_bp_n, den_bp_n = sig.lp2bp(num_lp_butter,den_lp_butter, wo=1, bw=1/.253)

# obtengo la transferencia desnormalizada del pasabanda
num_bp, den_bp = sig.lp2bp(num_lp_butter,den_lp_butter, wo=8.367, bw=33)

# Averiguo los polos y ceros
z_bp_n, p_bp_n, k_bp_n = sig.tf2zpk(num_bp_n, den_bp_n)

str_aux = 'Pasabanda normalizado'
print( str_aux )
print( '-' * len(str_aux) )

this_lti = sig.TransferFunction(num_bp_n, den_bp_n)
pretty_print_lti(this_lti)
print( '\n\n')

#%matplotlib qt5
# visualizo el modelo matemático normalizado
#analyze_sys([sig.TransferFunction(num_bp_n,den_bp_n)], ['mp'])
# o el modelo matemático desnormalizado
#analyze_sys([sig.TransferFunction(num_bp,den_bp)], ['mp'])

# visualización de cada sección

fig_id = 1
_, axes_hdl = bodePlot(sig.TransferFunction(num_bp_n,den_bp_n), label = 'Total', fig_id=fig_id )

this_k = k_bp_n
this_z = z_bp_n[0]
this_p = p_bp_n[0:2]
bodePlot(sig.TransferFunction(sig.lti(this_z,this_p,this_k)), label = 'Sect {:d}'.format(0), fig_id=fig_id, axes_hdl=axes_hdl )

str_aux = 'Section {:d}'.format(0)
print( str_aux )
print( '-' * len(str_aux) )

this_lti = sig.TransferFunction(sig.lti(this_z,this_p,this_k))
pretty_print_lti(this_lti)
print( '\n\n')

for ii in np.arange(1, p_bp_n.shape[0]/2, dtype='int'):
    
    this_k = 1
    this_z = z_bp_n[ii]
    this_p = p_bp_n[ii*2:ii*2+2]
    this_lti = sig.TransferFunction(sig.lti(this_z,this_p,this_k))

    str_aux = 'Section {:d}'.format(ii)
    print( str_aux )
    print( '-' * len(str_aux) )
    
    pretty_print_lti(this_lti)

    print( '\n\n')
    
    bodePlot(this_lti, label = 'Sect {:d}'.format(ii), fig_id=fig_id, axes_hdl=axes_hdl )
    

    
