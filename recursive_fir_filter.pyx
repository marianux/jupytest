#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:07:51 2023

@author: mariano
"""

import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
def MA_st_rt_calc_ci(np.ndarray[np.double_t, ndim=1] xx, int DD, int UU):
    cdef np.ndarray[np.double_t, ndim=1] hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)
    
    cdef np.ndarray[np.double_t, ndim=1] yy_ci = np.zeros(UU)
    
    yy_ci[:] = np.sum(xx[:(DD * UU)] * hh_u)
    
    cdef np.ndarray[np.double_t, ndim=1] xx_ci = xx[:(DD * UU)]

    return (xx_ci.copy(), yy_ci.copy())

@boundscheck(False)
@wraparound(False)
def MA_st_rt(np.ndarray[np.double_t, ndim=1] xx, int DD, int UU, np.ndarray[np.double_t, ndim=1] xx_ci, np.ndarray[np.double_t, ndim=1] yy_ci, int kk_offset=0):
    cdef int NN = xx.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] yy = np.zeros_like(xx)

    cdef np.ndarray[np.double_t, ndim=1] hh_u
    cdef int kk, ii

    if kk_offset == 0:
        for kk in range(UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy_ci[kk]
        yy[kk:DD * UU] = yy[kk]
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)
        for kk in range(DD * UU, (DD * UU) + UU):
            ii = kk - 1
            yy[ii] = np.sum(xx[kk-(DD * UU):kk] * hh_u)
    else:
        for kk in range(UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy_ci[kk]
        for kk in range(UU, DD * UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy[(kk - UU)]
        kk += 1

    for kk in range(kk, NN):
        yy[kk] = xx[kk] - xx[kk - DD * UU] + yy[kk - UU]

    xx_ci[:] = xx[(NN - DD * UU):]
    yy_ci[:] = yy[(NN - UU):]

    return (yy.copy()/DD, xx_ci.copy(), yy_ci.copy())


@boundscheck(False)
@wraparound(False)
def Tdc_seq_removal(np.ndarray xx, int DD=16, int UU=2, int MA_stages=2):
    cdef int NN = xx.shape[0]

    # Se calculan condiciones iniciales
    xx_ci, yy_ci = MA_st_rt_calc_ci(xx, DD, UU)

    cdef np.ndarray[np.double_t, ndim=1] yy = np.zeros_like(xx)

    # Supongo que block_s está definido en otro lugar o que es un valor conocido
    cdef int block_s = 10  # Define el tamaño de bloque según tus necesidades

    cdef int jj

    for jj in range(0, NN, block_s):
        yy_aux, xx_ci, yy_ci = MA_st_rt(xx[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
        yy[jj:jj+block_s] = yy_aux

    # Cascadeamos MA_stages-1 más
    cdef int ii
    for ii in range(1, MA_stages):
        # Se calculan condiciones iniciales
        xx_ci, yy_ci = MA_st_rt_calc_ci(yy, DD, UU)

        for jj in range(0, NN, block_s):
            yy_aux, xx_ci, yy_ci = MA_st_rt(yy[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
            yy[jj:jj+block_s] = yy_aux

    # Demora y resta
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU))
    yy = xx_aux - yy

    return yy
