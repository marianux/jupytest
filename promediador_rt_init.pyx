#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:28:15 2024

@author: mariano
"""

# promediador_rt_init.pyx

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

def promediador_rt_init(np.ndarray[double, ndim=1] xx, int DD, int UU):
    cdef int i
    cdef np.ndarray[double, ndim=1] hh_u = np.zeros(DD * UU, dtype=np.float64)
    
    for i in range(0, DD * UU, UU):
        hh_u[i] = 1
    hh_u = np.flip(hh_u)
    
    cdef np.ndarray[double, ndim=1] yy_ci = np.zeros(UU, dtype=np.float64)
    yy_ci[:] = np.sum(xx[:(DD * UU)] * hh_u)
    
    cdef np.ndarray[double, ndim=1] xx_ci = xx[:(DD * UU)]
    
    return xx_ci, yy_ci

