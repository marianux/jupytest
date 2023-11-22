#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:07:51 2023

@author: mariano
"""

# recursive_fir_filter.pyx

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def filter_sequence(int D, int U, double[:] x):
    
    cdef int N = x.shape[0]
    cdef int buffer_size = D * U
    cdef double[:] y = cython.view.array(shape=(N,), itemsize=sizeof(double), format="d")

    for k in range(N):

        # Calcula la salida según la ecuación recursiva
        y[k] = 1.0 / (D * U) * (x[k] - x[ (k - D * U) % buffer_size] + y[(k - U)% buffer_size])

    return y