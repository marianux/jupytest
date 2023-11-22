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
def filter_sequence(int N, int U, double[:] buffer, int[:] index, double x):
    # Agregar el nuevo valor al buffer
    buffer[index[0]] = x

    # Calcular la media cada U valores
    if index[0] % U == 0:
        average = sum(buffer) / (N * U)
    else:
        average = None

    # Actualizar el índice del buffer para el próximo valor
    index[0] = (index[0] + 1) % (N * U)

    return average

