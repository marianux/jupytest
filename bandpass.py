#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:14:49 2019

@author: mariano
"""

import scipy.signal as sig
import numpy as np
from splane import analyze_sys, pzmap, grpDelay, bodePlot

z,p,k = sig.buttap(2)
num_lp, den_lp = sig.zpk2tf(z,p,k)
#num_bp, den_bp = sig.lp2bp(num_lp,den_lp, wo=1, bw=1/.253)
num_bp, den_bp = sig.lp2bp(num_lp,den_lp, wo=8.367, bw=33)

#%matplotlib qt5
analyze_sys([sig.TransferFunction(num_bp,den_bp)], ['mp'])