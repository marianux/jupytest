#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:28:16 2024

@author: mariano
"""

# filtro_peine_DCyArmonicas.pyx
import numpy as np
cimport numpy as np
from promediador_rt_init import promediador_rt_init
from promediador_rt import promediador_rt

def filtro_peine_DCyArmonicas(np.ndarray[double, ndim=1] xx, int DD=16, int UU=2, int MA_stages=2, int block_s=1000):
    cdef int NN = xx.shape[0]
    cdef np.ndarray[double, ndim=1] xx_ci, yy_ci, yy_aux
    cdef np.ndarray[double, ndim=1] yy = np.zeros_like(xx)
    cdef int jj, ii
    
    # se calculan condiciones iniciales para el primer bloque moving averager (MA)
    # en total habrá MA_stages en cascada.
    xx_ci, yy_ci = promediador_rt_init( xx, DD, UU )
    
    # se procesa cada bloque por separado y se concatena la salida
    for jj in range(0, NN, block_s):
    
        yy_aux, xx_ci, yy_ci = promediador_rt( xx[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)

        yy[jj:jj+block_s] = yy_aux

    # cascadeamos MA_stages-1 más
    for ii in range(1, MA_stages):

        # se calculan condiciones iniciales
        xx_ci, yy_ci = promediador_rt_init( yy, DD, UU )
        
        for jj in range(0, NN, block_s):
        
            yy_aux, xx_ci, yy_ci = promediador_rt( yy[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
        
            yy[jj:jj+block_s] = yy_aux

    #############################################################
    # demora de la señal xx y resta de la salida del último MA
    
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU) )
    yy = xx_aux - yy
    return( yy )
