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
# z,p,k = sig.buttap(nn)
z,p,k = sig.besselap(nn, norm='delay')

num_lp, den_lp = sig.zpk2tf(z,p,k)

# paso de un Butter. a maxima planicidad si eps != 1
# num_lp_shift, den_lp_shift = sig.lp2lp(num_lp,den_lp, eps**(-1/nn) )
num_lp_shift, den_lp_shift = (num_lp,den_lp)



# obtengo la transferencia normalizada del pasabanda
num_bp_n, den_bp_n = sig.lp2bp(num_lp_shift,den_lp_shift, wo=1, bw=1/.253)

# obtengo la transferencia desnormalizada del pasabanda
num_bp, den_bp = sig.lp2bp(num_lp_shift,den_lp_shift, wo=8.367, bw=33)

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
this_lti = sig.TransferFunction(num_bp_n,den_bp_n)
_, axes_hdl = bodePlot(this_lti, label = 'Total', fig_id=fig_id )
all_sys = [this_lti]
str_labels = ['total'] 



## 
# Factorización en BP - BP
###########################

this_k = k_bp_n
this_z = z_bp_n[0]
this_p = p_bp_n[0:2]
this_lti = sig.TransferFunction(sig.lti(this_z,this_p,this_k))
bodePlot(this_lti, label = 'Sect {:d}'.format(0), fig_id=fig_id, axes_hdl=axes_hdl )
all_sys += [this_lti]

str_aux = 'Section {:d}'.format(0)
str_labels += [str_aux] 
print( str_aux )
print( '-' * len(str_aux) )

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
    
    all_sys += [this_lti] 
    str_labels += [str_aux] 
    

## 
# Factorización en LP - HP
###########################
    

# this_k = k_bp_n
# this_z = z_bp_n
# this_p = p_bp_n[0:2]
# this_lti = sig.TransferFunction(sig.lti(this_z,this_p,this_k))
# bodePlot(this_lti, label = 'HP', fig_id=fig_id, axes_hdl=axes_hdl )
# all_sys += [this_lti] 

# str_aux = 'Sección HP'
# str_labels += [str_aux] 
# print( str_aux )
# print( '-' * len(str_aux) )

# pretty_print_lti(this_lti)
# print( '\n\n')

# for ii in np.arange(1, p_bp_n.shape[0]/2, dtype='int'):
    
#     this_k = 1
#     this_z = []
#     this_p = p_bp_n[ii*2:ii*2+2]
#     this_lti = sig.TransferFunction(sig.lti(this_z,this_p,this_k))

#     str_aux = 'Sección LP'
#     print( str_aux )
#     print( '-' * len(str_aux) )
    
#     pretty_print_lti(this_lti)

#     print( '\n\n')

#     all_sys += [this_lti] 
#     str_labels += [str_aux] 
    

analyze_sys(all_sys, str_labels)
    
    
