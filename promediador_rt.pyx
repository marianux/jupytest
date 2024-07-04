#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:28:16 2024

@author: mariano
"""

# promediador_rt.pyx

import numpy as np
cimport numpy as np

def promediador_rt(np.ndarray[double, ndim=1] xx, int DD, int UU, 
                   np.ndarray[double, ndim=1] xx_ci, 
                   np.ndarray[double, ndim=1] yy_ci, int kk_offset=0):
    cdef int NN = xx.shape[0]
    cdef np.ndarray[double, ndim=1] yy = np.zeros_like(xx)
    cdef int kk, ii
    cdef np.ndarray[double, ndim=1] hh_u
    cdef np.ndarray[double, ndim=1] bb

    if(kk_offset == 0):

        # condiciones iniciales
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy_ci[kk]
              
        # extiendo las salidas al mismo valor que yy[UU]

        yy[kk:DD * UU] = yy[kk]
        
        # vector para filtrar muestras
        bb = np.zeros(DD * UU)
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)

        # inicio de la recursión
        for kk in range(DD * UU, (DD * UU) + UU ):
    
            ii = kk-1
            # Calcula la salida según la ecuación recursiva
            aa = kk-(DD * UU)
            bb = xx[aa:kk]
            print('tamaños aa:{:d} kk:{:d} len(bb):{:d} '.format(aa, kk, len(bb)))
            yy[ii] = np.sum(bb * hh_u)

    else:
        # para todos los bloques restantes salvo el primero
           
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy_ci[kk]
        
        for kk in range(UU, DD * UU):

            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy[(kk - UU)]
    
        #
        kk += 1

    # for kk in range(NN):
    for kk in range(kk, NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk] = xx[kk]  \
                  - xx[kk - DD * UU] \
                  + yy[kk - UU]
    
    # calculo las condiciones iniciales del siguiente bloque
    xx_ci = xx[(NN - DD * UU):]
    yy_ci = yy[(NN - UU):]

    # escalo y devuelvo
    return( (yy.copy()/DD, xx_ci.copy(), yy_ci.copy()) )
