#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:19:00 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import analyze_sys, pzmap, grpDelay, bodePlot


z,p,k = sig.buttap(3)
num_lp, den_lp = sig.zpk2tf(z,p,k)
eps = np.sqrt(10**(0.1)-1)

num_lp_d, den_lp_d = sig.lp2lp(num_lp,den_lp, eps**(-1/3) )
num_hp_d, den_hp_d = sig.lp2hp(num_lp_d, den_lp_d)

num_hp_d, den_hp_d = sig.lp2hp(num_lp,den_lp, eps**(-1/3) )

#%matplotlib qt5
analyze_sys([sig.TransferFunction(num_lp, den_lp)], ['mp_norm'])
analyze_sys([sig.TransferFunction(num_lp_d,den_lp_d)], ['mp_desnorm'])
