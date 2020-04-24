#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import tfadd, tfcascade, analyze_sys, pzmap, grpDelay, bodePlot, pretty_print_lti


nn = 2 # orden
ripple = 3 # dB


eps = np.sqrt(10**(ripple/10)-1)

z,p,k = sig.besselap(nn, norm='delay')
# z,p,k = sig.buttap(nn)


num_lp, den_lp = sig.zpk2tf(z,p,k)
num_lp, den_lp = sig.lp2lp(num_lp, den_lp, eps**(-1/nn) )

num_hp, den_hp = sig.lp2hp(num_lp,den_lp)
lp_sys = sig.TransferFunction(num_lp,den_lp)
hp_sys = sig.TransferFunction(num_hp,den_hp)
xover = tfadd(lp_sys, hp_sys)
bandpass = tfcascade(lp_sys, hp_sys)

pretty_print_lti(lp_sys)
pretty_print_lti(hp_sys)
pretty_print_lti(xover)
pretty_print_lti(bandpass)


analyze_sys([lp_sys, hp_sys, xover, bandpass], ['lp', 'hp', 'xover', 'bandpass'])

