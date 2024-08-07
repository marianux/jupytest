#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:51:39 2024

@author: mariano
"""

import numpy as np

def LSE_FIR(Norder, Be, D):

    #Modification of Burrus first algorithm

    N = Norder + 1
    Be = Be / np.pi
    h = fir3(N, Be, D)


def fir3(N, f, m):

    #h = fir3(N, f, m) calculates an optimal least square error
    #multiband FIR filter from a simple lowpass design.

    #f and m must  be the same even length. m must have pairs
    #of equal values (the ideal filter must be a pice-wise constant).
    #Remember that an even length filter must be zero at f = 1.
    #The multiband filter is constructed from lowpass filters
    #designed by fir3lp.m

    L = len(f)  #Number of band edges
    if m[-1] == 0:
        h = np.zeros(1, N)  #Frequency response is zero at pi
    else:
        h = np.concatenate([np.zeros(1, (N - 1) // 2), m[-1], np.zeros(1, (N - 1) // 2)])
    while L > 2:
        h = h + (m[L - 2] - m[L - 1]) * fir3lp(N, f[L - 2], f[L - 1])  #Construct
        L = L - 2


def fir3lp(N, f1, f2, p=None):

    #b = fir3lp(N, f1, f2, p) designs a linear phase lowpass FIR filter
    #b(n) of length N with a least integral squared error approximation
    #to an ideal lowpass filter with a passband from 0 to f1 and a stopband
    #from f2 to 1. (in normalized Hertz) and a p-order spline transition
    #band. If p is not given, a near optimal value is calculated as
    #p = 0.62*N*d.

    if p is None:
        p = 0.31 * N * (f2 - f1)  #Optimal spline power p

    w0 = np.pi * (f2 + f1) / 2  #Average band edge
    dw = np.pi * (f2 - f1) / 2  #Half transition width
    if N % 2 == 0:
        nx = np.arange(1, (N - 1) // 2 + 1)  #Even length index vector
    else:
        nx = np.arange(1, (N - 1) // 2 + 1)  #Odd length index vector
    M = len(nx)
    h = np.sin(w0 * nx) / (np.pi * nx)  #LP filter with no transition
    if dw != 0 and p != 0:  #p-order spline transition fn
        wd = (dw / p) * nx  #Weighing function: wt
        wt = (np.sin(wd) / wd)**p
        h = wt * h
    if N % 2 == 0:
        b = np.concatenate([h[M - 1: -1], h])  #Even length output
    else:
        b = np.concatenate([h[M - 1: -1], w0 / np.pi, h])  #Odd length output
