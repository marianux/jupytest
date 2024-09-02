#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:58:40 2024

@author: mariano
"""

import numpy as np
import warnings

def REMEZ_EX(Nfilt, Neg, Nfcns, Ngrid, Grid, Iext, Des, Wt):
    """
    Implements the Remez exchange algorithm for the weighted Chebyshev 
    approximation of a continuous function with a sum of cosines.

    Parameters:
    Nfilt (int): Filter order
    Neg (int): Negative frequencies flag
    Nfcns (int): Number of filter coefficients
    Ngrid (int): Number of grid points
    Grid (numpy array): Frequency grid between 0 and 1
    Iext (numpy array): Initial extremal frequencies
    Des (numpy array): Desired frequency response
    Wt (numpy array): Weight function

    Returns:
    h (numpy array): Coefficients of the filter
    Err (float): The resulting value of the weighted error function
    Dev (float): Deviation from the desired function
    """
    
    Nodd = Nfilt % 2
    Nm1 = Nfcns - 1
    Tpi = 2 * np.pi
    goto = 1

    while goto <= 43:
        if goto == 1:
            Itrmax = 2500
            Devl = -1
            Nzr = Nfcns + 1
            Nzz = Nfcns + 2
            Niter = 0
            goto = 2
        elif goto == 2:
            Iext[Nzz] = Ngrid + 1
            Niter += 1
            if Niter > Itrmax:
                break
            X = np.cos(Tpi * Grid[Iext[:Nzr]])
            Jet = (Nfcns - 1) // 15 + 1
            Ad = np.array([D(j, Nzr, Jet, X) for j in range(Nzr)])
            Dnum = Dden = 0
            Kr = 1
            for j in range(Nzr):
                Lr = Iext[j]
                Dnum += Ad[j] * Des[Lr]
                Dden += Kr * Ad[j] / Wt[Lr]
                Kr = -Kr
            Dev = Dnum / Dden
            Nu = -1 if Dev > 0 else 1
            Dev = -Nu * Dev
            Kr = Nu
            Y = np.array([Des[Iext[j]] + Kr * Dev / Wt[Iext[j]] for j in range(Nzr)])
            Kr = -Kr
            if Dev < Devl:
                print(" ****** Failure to converge ******** ")
                print(" 1 Probable cause is machine rounding error.")
                print(" 2 Impulse response may be correct.")
                print(" 3 Check with a frequency response analysis.")
                goto = 50
            Devl = Dev
            Jchange = 0
            K1 = Iext[0]
            Knz = Iext[Nzr - 1]
            Klow = 0
            Nut = -Nu
            Jr = 1
            goto = 5
        elif goto == 5:
            if Jr == Nzz:
                Ynz = Comp
            if Jr >= Nzz:
                if Jr > Nzz:
                    if Luck > 9:
                        Kn = Iext[Nzz - 1]
                        Iext[:Nfcns] = Iext[1:Nfcns + 1]
                        Iext[Nzr - 1] = Kn
                        goto = 2
                    else:
                        if Comp > Y1:
                            Y1 = Comp
                        K1 = Iext[Nzz - 1]
                        goto = 36
                else:
                    if K1 > Iext[0]:
                        K1 = Iext[0]
                    if Knz < Iext[Nzr - 1]:
                        Knz = Iext[Nzr - 1]
                    Nut1 = Nut
                    Nut = -Nu
                    Lr = 0
                    Kupr = K1
                    Comp = Ynz * 1.0001
                    Luck = 1
                    goto = 30
            else:
                goto = 6
        elif goto == 6:
            Kupr = Iext[Jr]
            Lr = Iext[Jr - 1] + 1
            Nut = -Nut
            if Jr == 2:
                Y1 = Comp
            Comp = Dev
            if Lr >= Kupr:
                goto = 13
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 13
                else:
                    Comp = Nut * Err
                    goto = 9
        elif goto == 9:
            Lr += 1
            if Lr >= Kupr:
                goto = 12
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 12
                else:
                    Comp = Nut * Err
                    goto = 9
        elif goto == 12:
            Iext[Jr - 1] = Lr - 1
            Jr += 1
            Klow = Lr - 1
            Jchange += 1
            goto = 5
        elif goto == 13:
            Lr -= 1
            goto = 14
        elif goto == 14:
            Lr -= 1
            if Lr <= Klow:
                Lr = Iext[Jr - 1] + 1
                if Jchange > 0:
                    goto = 12
                else:
                    goto = 24
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp > 0:
                    Comp = Nut * Err
                    goto = 19
                else:
                    if Jchange <= 0:
                        goto = 14
                    else:
                        goto = 27
        elif goto == 19:
            Lr -= 1
            if Lr <= Klow:
                goto = 22
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 22
                else:
                    Comp = Nut * Err
                    goto = 19
        elif goto == 22:
            Klow = Iext[Jr - 1]
            Iext[Jr - 1] = Lr + 1
            Jr += 1
            Jchange += 1
            goto = 5
        elif goto == 24:
            Lr += 1
            if Lr >= Kupr:
                goto = 27
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 24
                else:
                    Comp = Nut * Err
                    goto = 9
        elif goto == 27:
            Klow = Iext[Jr - 1]
            Jr += 1
            goto = 5
        elif goto == 30:
            Lr += 1
            if Lr >= Kupr:
                Luck = 6
                goto = 36
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 30
                else:
                    Comp = Nut * Err
                    Jr = Nzz
                    goto = 9
        elif goto == 36:
            Lr = Ngrid + 1
            Klow = Knz
            Nut = -Nut1
            Comp = Y1 * 1.00001
            goto = 37
        elif goto == 37:
            Lr -= 1
            if Lr <= Klow:
                if Luck == 6:
                    if Jchange > 0:
                        goto = 2
                    else:
                        goto = 50
                else:
                    Iext[:Nfcns] = Iext[Nzr - 1::-1]
                    Iext[0] = K1
                    goto = 2
            else:
                Geex = Gee(Lr, Nzr, Ad, Grid, X, Y)
                Err = (Geex - Des[Lr]) * Wt[Lr]
                Dtemp = Nut * Err - Comp
                if Dtemp <= 0:
                    goto = 37
                else:
                    Jr = Nzz
                    Comp = Nut * Err
                    Luck += 10
                    goto = 19

    # Final calculation of coefficients of the best approximation
    Fsh = 0.000001
    Gtemp = Grid[0]
    X[Nzz - 1] = -2
    Cn = 2 * Nfcns - 1
    Delfr = 1 / Cn
    Lr = 1
    Kkk = 0
    if Grid[0] == 0 and Grid[Ngrid - 1] == 0.5:
        Kkk = 1
    if Nfcns <= 3:
        Kkk = 1
    if Kkk != 1:
        Dtemp = np.cos(Tpi * Grid[0])
        Dnum = np.cos(Tpi * Grid[Ngrid - 1])
        Aa = 2 / (Dtemp - Dnum)
        Bb = -(Dtemp + Dnum) / (Dtemp - Dnum)
    for Jr in range(1, Nfcns + 1):
        Ft = (Jr - 1) * Delfr
        Xt = np.cos(Tpi * Ft)
        if Kkk != 1:
            Xt = (Xt - Bb) / Aa
            Ft = np.arccos(Xt) / Tpi
        flag = 0
        while flag == 0:
            Xe = X[Lr]
            if Xt > Xe:
                flag = 1
                if Xt - Xe < Fsh:
                    A = Y[Lr]
                else:
                    Grid[0] = Ft
                    Geex = Gee(0, Nzr, Ad, Grid, X, Y)
                    A = Geex
            else:
                if Xe - Xt < Fsh:
                    A = Y[Lr]
                    flag = 1
                else:
                    Lr += 1
        if Lr > 1:
            Lr -= 1
    Grid[0] = Gtemp
    Dden = Tpi / Cn
    Alpha = np.zeros(Nfcns)
    for Jr in range(1, Nfcns + 1):
        Dtemp = 0
        Dnum = (Jr - 1) * Dden
        if Nm1 >= 1:
            for Kr in range(1, Nm1 + 1):
                Dtemp += A[Kr] * np.cos(Dnum * Kr)
        Alpha[Jr - 1] = 2 * Dtemp + A[0]
    for Jr in range(2, Nfcns + 1):
        Alpha[Jr - 1] = 2 * Alpha[Jr - 1] / Cn
    Alpha[0] /= Cn
    if Kkk != 1:
        P = np.zeros(Nfcns)
        P[0] = 2 * Alpha[Nfcns - 1] * Bb + Alpha[Nm1 - 1]
        P[1] = 2 * Aa * Alpha[Nfcns - 1]
        Q = np.zeros(Nm1)
        Q[0] = Alpha[Nm1 - 2] - Alpha[Nfcns - 1]
        for Jr in range(2, Nm1 + 1):
            if Jr >= Nm1:
                Aa *= 0.5
                Bb *= 0.5
            P[Jr] = 0
            A[:Jr] = P[:Jr]
            P[:Jr] = 2 * Bb * A[:Jr]
            P[1] += A[0] * 2 * Aa
            Jm1 = Jr - 1
            for Kr in range(1, Jm1 + 1):
                P[Kr] += Q[Kr] + Aa * A[Kr]
            Jp1 = Jr + 1
            for Kr in range(2, Jp1):
                P[Kr] += Aa * A[Kr - 1]
            if Jr != Nm1:
                Q[:Jr] = -A[:Jr]
                Q[0] += Alpha[Nfcns - 1 - Jr]
        Alpha = P
    if Nfcns <= 3:
        Alpha[Nfcns] = 0
        Alpha[Nfcns + 1] = 0

    h = Impulse(Neg, Nodd, Nfcns, Alpha)
    return h, Dev, Iext


def D(k, N, M, X):
    """ Calculate the Lagrange interpolation coefficients """
    D = 1
    Q = X[k]
    for L in range(M):
        for j in range(L, N, M):
            if j != k:
                D *= 2 * (Q - X[j])
    return 1 / D

def gee(L, N, Ad, Grid, X, Y):
    """ Evaluate the frequency response using Lagrange interpolation in Barycentric form """
    Tpi = 2 * np.pi
    P = 0
    D = 0
    for j in range(N):
        C = np.cos(Tpi * Grid[L]) - X[j]
        C = Ad[j] / C
        D += C
        P += C * Y[j]
    return P / D

def impulse(Neg, Nodd, Nfcns, Alpha):
    """ Calculate the impulse response """
    Nm1 = Nfcns - 1
    Nz = Nfcns + 1
    h = np.zeros(Nfcns + 1)
    
    if Neg == 1:
        if Nodd == 0:
            h[0] = 0.25 * Alpha[Nfcns - 1]
            for j in range(1, Nm1):
                h[j] = 0.25 * (Alpha[Nz - j - 1] - Alpha[Nfcns - j - 1])
            h[Nfcns - 1] = 0.5 * Alpha[0] - 0.25 * Alpha[1]
        else:
            h[0] = 0.25 * Alpha[Nfcns - 1]
            h[1] = 0.25 * Alpha[Nm1 - 1]
            for j in range(2, Nm1):
                h[j] = 0.25 * (Alpha[Nz - j - 1] - Alpha[Nfcns - j])
            h[Nfcns - 1] = 0.5 * Alpha[0] - 0.25 * Alpha[2]
            h[Nz - 1] = 0
    else:
        if Nodd == 0:
            h[0] = 0.25 * Alpha[Nfcns - 1]
            for j in range(1, Nm1):
                h[j] = 0.25 * (Alpha[Nz - j - 1] + Alpha[Nfcns - j - 1])
            h[Nfcns - 1] = 0.5 * Alpha[0] + 0.25 * Alpha[1]
        else:
            for j in range(Nm1):
                h[j] = 0.5 * Alpha[Nz - j - 1]
            h[Nfcns - 1] = Alpha[0]
    
    return h


def REMEZ_FIR(order, edge, fx, *args):
    """
    REMEZ_FIR - A translation of the FORTRAN code of the Parks-McClellan
    minimax arbitrary-magnitude FIR filter design algorithm into Python.
    """
    edge = np.array(edge) / np.pi

    lgrid = 16  # default value
    nn = len(args)
    if nn > 0 and isinstance(args[-1], list):
        lgrid = args[-1][0]
        nn -= 1
        args = args[:nn]

    if nn == 1:
        if isinstance(args[0], str):
            filter_type = args[0]
            wtx = np.ones(len(fx) // 2)
        else:
            wtx = np.array(args[0])
            filter_type = 'multiband'
    elif nn == 2:
        wtx = np.array(args[0])
        filter_type = args[1]
    else:
        wtx = np.ones(len(fx) // 2)
        filter_type = 'multiband'

    # Determine filter type
    if filter_type.lower().startswith('d'):
        jtype = 2  # Differentiator
    elif filter_type.lower().startswith('h'):
        jtype = 3  # Hilbert transformer
    elif filter_type.lower().startswith('m'):
        jtype = 1  # Multiband filter
    else:
        jtype = 1  # Multiband filter

    # Determine the filter cases and nfcns
    if jtype == 1:
        if order % 2 == 0:
            filtercase = 1  # Even order and even symmetry
        else:
            filtercase = 2  # Odd order and even symmetry
    else:
        if order % 2 == 0:
            filtercase = 3  # Even order and odd symmetry
        else:
            filtercase = 4  # Odd order and odd symmetry

    nfcns = (order + 1) // 2
    if filtercase == 1:
        nfcns += 1

    # Determine grid, des, and wt
    nbands = len(edge) // 2
    delf = 0.5 / (lgrid * nfcns)
    delf *= 2
    grid = []
    des = []
    wt = []

    for ll in range(nbands):
        number_grid = int(np.ceil((edge[2 * ll + 1] - edge[2 * ll]) / delf))
        grid_more = np.linspace(edge[2 * ll], edge[2 * ll + 1], number_grid + 1)
        
        # Adjust grid for harmful frequencies
        if ll == 0 and (filtercase == 3 or filtercase == 4) and grid_more[0] < delf:
            grid_more = grid_more[1:]
            number_grid -= 1

        if ll == nbands - 1 and (filtercase == 2 or filtercase == 3) and grid_more[-1] > 1 - delf:
            grid_more = grid_more[:-1]
            number_grid -= 1

        grid.extend(grid_more)

        if jtype != 2:
            wt_more = wtx[ll] * np.ones(number_grid + 1)
            if fx[2 * ll + 1] != fx[2 * ll]:
                des_more = np.linspace(fx[2 * ll], fx[2 * ll + 1], number_grid + 1)
            else:
                des_more = fx[2 * ll] * np.ones(number_grid + 1)
        else:
            des_more = fx[2 * ll] * grid_more * np.pi
            if abs(fx[2 * ll]) < 1.0e-3:
                wt_more = wtx[ll] * np.ones(number_grid + 1)
            else:
                wt_more = wtx[ll] / (grid_more * np.pi)

        des.extend(des_more)
        wt.extend(wt_more)

    grid = np.array(grid)
    des = np.array(des)
    wt = np.array(wt)

    # Modify des and wt depending on the filter case
    if filtercase == 2:
        des /= np.cos(np.pi * grid / 2)
        wt *= np.cos(np.pi * grid / 2)
    if filtercase == 4:
        des /= np.sin(np.pi * grid / 2)
        wt *= np.sin(np.pi * grid / 2)
    if filtercase == 3:
        des /= np.sin(np.pi * grid)
        wt *= np.sin(np.pi * grid)

    # Call the REMEZ algorithm
    h, err, iext = REMEZ_EX_A(nfcns, grid, des, wt)

    # Generate the impulse responses for other types
    nn = len(h)
    if filtercase == 2:
        h = np.concatenate(([h[0] / 2], (h[1:] + h[:-1]) / 2, [h[-1] / 2]))
    if filtercase == 3:
        h = np.concatenate(([h[0] / 2, h[1] / 2], (h[2:] - h[:-2]) / 2, [-h[-2] / 2, -h[-1] / 2]))
    if filtercase == 4:
        h = np.concatenate(([h[0] / 2], (h[1:] - h[:-1]) / 2, [-h[-1] / 2]))

    err = abs(err)
    return h, err


def remez_ex_a(nfcns, grid, des, wt):
    """
    Implements the Remez exchange algorithm for the weighted Chebyshev 
    approximation of a continuous function with a sum of cosines.

    Parameters:
    nfcns (int): Number of basis functions
    grid (numpy array): Frequency grid between 0 and 1
    des (numpy array): Desired function on frequency grid
    wt (numpy array): Weight function on frequency grid

    Returns:
    h (numpy array): Coefficients of the filter
    dev (float): The resulting value of the weighted error function
    iext (numpy array): Indices of extremal frequencies
    """
    
    # Initializations
    ngrid = len(grid)
    l_ove = np.arange(1, ngrid + 1)
    temp = (ngrid - 1) / nfcns
    jj = np.arange(1, nfcns + 1)
    l_trial = np.concatenate((np.fix(temp * (jj - 1) + 1), [ngrid])).astype(int)
    nz = nfcns + 1
    devl = 0
    niter = 1
    itrmax = 250
    x_all = np.cos(np.pi * grid)
    
    # Remez loop
    while niter < itrmax:
        x = np.cos(np.pi * grid[l_trial - 1])
        
        # Calculate the Lagrange interpolation coefficients
        jet = (nfcns - 1) // 15 + 1
        ad = np.zeros(nz)
        
        for mm in range(nz):
            yy = 1
            for nn in range(jet):
                xx = 2 * (x[mm] - x[nn:jet:nz])
                yy *= np.prod(xx[xx != 0])
            ad[mm] = 1 / yy
        
        alter = np.ones_like(ad)
        alter[1::2] = -alter[1::2]
        dnum = np.dot(ad, des[l_trial - 1])
        dden = np.dot(alter, ad / wt[l_trial - 1])
        dev = -dnum / dden
        
        if abs(dev) <= abs(devl):
            warnings.warn('Convergence problems')
            break
        
        devl = dev
        y = des[l_trial - 1] + dev * alter / wt[l_trial - 1]
        l_left = np.setdiff1d(l_ove, l_trial)
        err_num = np.zeros_like(l_left, dtype=float)
        err_den = np.zeros_like(l_left, dtype=float)
        
        for jj in range(nz):
            aid = ad[jj] / (x_all[l_left - 1] - x[jj])
            err_den += aid
            err_num += y[jj] * aid
        
        wei_err = (err_num / err_den - des[l_left - 1]) * wt[l_left - 1]
        wei_err = np.append(wei_err, alter * dev)
        l_aid1 = np.where(np.diff(np.sign(np.diff(np.concatenate(([0], wei_err, [0])))))[0])[0] + 1
        l_aid2 = l_aid1[np.abs(wei_err[l_aid1]) >= np.abs(dev)]
        
        if len(l_aid2) > 0:
            X, ind = max(enumerate(np.abs(wei_err[l_aid2])), key=lambda x: x[1])
            l_real_init = l_aid2[ind]
        else:
            l_real_init = []
        
        if len(l_real_init) % 2 == 1:
            if abs(wei_err[l_real_init[0]]) <= abs(wei_err[l_real_init[-1]]):
                l_real_init = l_real_init[1:]
            else:
                l_real_init = l_real_init[:-1]
        
        while len(l_real_init) > nz:
            wei_real = np.abs(wei_err[l_real_init])
            wei_comp = np.maximum(wei_real[:-1], wei_real[1:])
            if max(abs(wei_err[l_real_init[0]]), abs(wei_err[l_real_init[-1]])) <= min(wei_comp):
                l_real_init = l_real_init[1:-1]
            else:
                ind_omit = np.argmin(wei_comp)
                l_real_init = np.delete(l_real_init, [ind_omit, ind_omit + 1])
        
        l_real = l_real_init
        
        if np.array_equal(l_real, l_trial):
            break
        else:
            l_trial = l_real
            niter += 1
    
    iext = l_real
    
    # Generate the impulse response using the IDFT
    cn = 2 * nfcns - 1
    x_IDFT = np.cos(np.pi * np.arange(0, 2 * (nfcns - 1) + 1) / cn)
    ind1, ind2 = np.intersect1d(x_IDFT, x, return_indices=True)
    ind1 = np.sort(ind1)
    ind2 = np.sort(ind2)
    
    l_left = np.setdiff1d(np.arange(1, nfcns + 1), ind1 + 1)
    num = np.zeros(len(l_left))
    den = np.zeros(len(l_left))
    
    for jj in range(nz):
        aid = ad[jj] / (x_IDFT[l_left - 1] - x[jj])
        den += aid
        num += y[jj] * aid
    
    A = np.zeros_like(x_IDFT)
    A[l_left - 1] = num / den
    A[ind1] = y[ind2]
    
    h = np.zeros(nfcns)
    
    for n in range(1, nfcns + 1):
        h[n - 1] = (1 / cn) * (A[0] + 2 * np.sum(A[1:nfcns] * np.cos(2 * np.pi * np.arange(1, nfcns) * (n - 1) / cn)))
    
    h = np.real(h)
    h = np.concatenate((h[::-1], h[1:]))
    
    return h, dev, iext


def HERRMANN_LP_FIR_ORDER(wT, d):
    """
    Herrmann's estimate of the required order, N, for minimax designed  
    linear-phase lowpass FIR filters.
    """
    if wT[0] >= wT[1] or wT[0] >= np.pi or wT[1] > np.pi:
        raise ValueError("Improper band edges. Edges should be: wcT < wsT <= π.")

    wcT = wT[0]  # Passband edge
    wsT = wT[1]  # Stopband edge
    dwT = wsT - wcT  # Transition band
    dc = d[0]
    ds = d[1]

    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -0.4761
    a4 = -2.66e-3
    a5 = -0.5941
    a6 = -0.4278
    b1 = 11.01217
    b2 = 0.51244

    ldc = np.log10(dc)
    lds = np.log10(ds)

    if dc < ds:
        ldc, lds = lds, ldc  # Swap if dc is smaller

    F = b1 + b2 * (ldc - lds)
    D = (a1 * ldc**2 + a2 * ldc + a3) * lds + (a4 * ldc**2 + a5 * ldc + a6)
    Tpi = 2 * np.pi
    N = np.ceil(Tpi * D / dwT - F * dwT / Tpi)

    Be = np.array([0, wcT, wsT, np.pi])
    D = np.array([1, 1, 0, 0])
    W = np.array([1, d[0] / d[1]])

    return int(N), Be, D, W


def L_PHASE_LP_HB_FIR_ORDER(wsT, ds):
    """
    Computes the order of an even-order, half-band, equiripple,  
    linear-phase lowpass FIR filter.
    """
    N, Be, D, W = HERRMANN_LP_FIR_ORDER([np.pi - wsT, wsT], [ds, ds])
    N = 4 * round((N - 1) / 4) + 2  # Ensure it is a half-band filter
    return N


def L_PHASE_BP_FIR_ORDER(wT_edge, d):
    """
    Estimates the required order for a linear-phase bandpass FIR filter.
    """
    if not (wT_edge[0] < wT_edge[1] < wT_edge[2] < wT_edge[3]):
        raise ValueError("Improper band edges. Edges should be: ws1T < wc1T < wc2T < ws2T.")

    DwT = min(wT_edge[1] - wT_edge[0], wT_edge[3] - wT_edge[2]) / (2 * np.pi)
    DwTmax = max(wT_edge[1] - wT_edge[0], wT_edge[3] - wT_edge[2]) / (2 * np.pi)
    R = DwTmax / DwT
    xc = np.log10(d[1])
    xs = np.log10(min(d[0], d[2]))

    b1 = 0.01201
    b2 = 0.09664
    b3 = -0.51325
    b4 = 0.00203
    b5 = -0.57054
    b6 = -0.44314

    C = ((b1 * xc + b2) * xc + b3) * xs + (b4 * xc + b5) * xc + b6
    G = 14.6 * (xc - xs) + 16.9
    N = int(np.floor((C / DwT + G * DwT + 1) * (1 - 0.034 * (R - 2))))

    Be = np.array([0, wT_edge[0], wT_edge[1], wT_edge[2], wT_edge[3], np.pi])
    D = np.array([0, 0, 1, 1, 0, 0])
    W = np.array([d[1] / d[0], 1, d[1] / d[2]])

    return N, Be, D, W


def L_PHASE_LP_FIR_ORDER(wT, d):
    """
    Estimates the required order, N, for minimax designed linear-phase 
    lowpass FIR filters.
    """
    if wT[0] >= wT[1] or wT[0] >= np.pi or wT[1] > np.pi:
        raise ValueError("Improper band edges. Edges should be: wcT < wsT <= π.")

    fc = wT[0] / (2 * np.pi)  # Normalize to half the sampling frequency
    fs = wT[1] / (2 * np.pi)
    deltaF = fs - fc
    deltac = d[0]
    deltas = d[1]
    N = N4(0.5 - fc, deltaF, deltac, deltas) - 1

    Be = np.array([0.0, wT[0], wT[1], np.pi])
    D = np.array([1, 1, 0, 0])
    W = np.array([1, d[0] / d[1]])

    return N, Be, D, W


def h13(fc, deltaF, c):
    return (2 / np.pi) * np.arctan((c / deltaF) * (1 / fc - 1 / (1 - deltaF)))


def g13(fc, deltaF, delta):
    v = 2.325 * (-np.log10(delta))**-0.445 * deltaF**-1.39
    return (2 / np.pi) * np.arctan(v * (1 / fc - 1 / (1 - deltaF)))


def Nc(deltaF, delta):
    return int(np.ceil((1.101 / deltaF) * (-np.log10(2 * delta))**1.1 + 1))


def N3(fc, deltaF, b):
    return int(np.ceil(Nc(deltaF, b) * (g13(fc, deltaF, b) + g13(1 - deltaF - fc, deltaF, b) + 1) / 3))


def Nm(deltaF, deltac, deltas):
    return (0.52 / deltaF) * np.log10(deltac / deltas) * (-np.log10(deltac))**0.17


def DN(fc, deltaF, deltac, deltas):
    return int(np.ceil(Nm(deltaF, deltac, deltas) * (h13(fc, deltaF, 1.1) - 0.5 * (h13(1 - deltaF - fc, deltaF, 0.29) - 1))))


def N4(fc, deltaF, deltac, deltas):
    return N3(fc, deltaF, deltac) + DN(fc, deltaF, deltac, deltas)



# Define the band edges and gains
wTedges = np.array([0.3, 0.6]) * np.pi
b = [1, 0]  # Gains in the bands
d = [0.02, 0.0025]  # Acceptable deviations

# Compute the required stopband attenuation and passband ripple
Amaxreq = 20 * np.log10(1 + 0.02 / (1 - 0.02))
Aminreq = 20 * np.log10((1 + 0.02) / 0.0025)

print(f"Amaxreq = {Amaxreq:.4f} dB")
print(f"Aminreq = {Aminreq:.4f} dB")

# Estimate filter order
N, Be, D, W = L_PHASE_LP_FIR_ORDER(wTedges, d)
W = [1, 4]  # Weighting factors

# Output estimated order
print(f"Estimated order (N): {N}")

# Increase order to ensure meeting the criteria
N += 1  # Minimum order (16) to meet the criteria
print(f"Final filter order: {N}")

# Filter type (multiband)
Ftype = 'm'

# Design the filter using the Remez algorithm
h, Err = REMEZ_FIR(N, Be, D, W, Ftype, [128])

# Calculate resulting passband and stopband ripples
deltac = Err / W[0]
deltas = Err / W[1]

# Normalize the maximum gain to 1
hn = h / (1 + deltac)

# Output the errors and attenuation values
print(f"Err = {Err:.8f}")
Amax = 20 * np.log10((1 + deltac) / (1 - deltac))
Amin = 20 * np.log10((1 + deltac) / deltas)
print(f"Amax = {Amax:.6f} dB")
print(f"Amin = {Amin:.4f} dB")
