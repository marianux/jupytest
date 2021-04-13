#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import tfadd, tfcascade, analyze_sys, pzmap, grpDelay, bodePlot, pretty_print_lti

kn = 1

# num
qn = 1e6
wn = 2
# num
qd = np.sqrt(2)/2
wd = 1

tf_bicuad = sig.TransferFunction(kn*np.array([1, wn/qn, wn**2]), np.array([1, wd/qd, wd**2]))


pretty_print_lti(tf_bicuad)


analyze_sys([tf_bicuad], ['bicuad'])

