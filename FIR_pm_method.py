#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:58:40 2024

@author: mariano
"""

import numpy as np
import warnings
from scipy.sparse import coo_matrix 

from scipy.signal import remez
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def REMEZ_FIR_MLLS(order, edge, fx, lgrid = 16, fs = 2.0, 
              filter_type = 'multiband', wtx = None, maxiter = 250, bDebug=False):
    """
    RERMEZ_FIR - A translation of the FORTRAN code of the Parks-McClellan
    minimax arbitrary-magnitude FIR filter design algorithm [1], referred 
    in later use to as the PMR code, into MATLAB (by Tapio Saramaki, 
    2018-02-20), and then into Python by MLLS.
    	
    Compared with the PMR code, the the peculiarities of the present code are: 
    (1) The Remez loop has been improved according to [2].
    (2) Due to the use of MATLAB, the grid pont allocation happened to be 
        improved.
    (3) The desired function is given in a slightly different manner.
    	
    Inputs to be always used:
    order - filter order
    edge  - specifies the upper and lower edges of the bands under consideration.
            The program, however, uses band edes in terms of fractions of pi rad.
    	    edge = edge/pi;
    fx -    specifies the desired values at the edges of each band.
    	
    Inputs to a possible inclusion in varargin: 
    wtx   - a constant weighting in each band
    type  - three filter types are under consideration, namely,
            (1) Filters with symmetric impulse response for multiband design: 
                lowpass, higpass, bandpass, and bandstop designs are typical
                examples. The benefit of these filters is that the phase is
                perfectly linear for all frequencies.
            (2) Full band and partial band differentiators: filters with 
                antisymmetric impulse response are in use.
            (3) Hilbert transformers: again antisymmetry is required.
            The distinction between these three types is performed by a 
            character array, called type, such that
            type(1)='m' or type(1)='M' ==> 'multiband' design
            type(1)='d' or type(1)='D' ==> 'differentiator'
            type(1)='h' or type(1)='H' ==> 'Hilbert' tranformer
            If type is not specified, then type=='multiband' is used
    lgrid - if not specified, lgrid=16 as in the PMR code. To  increase the 
            accuray of the resulting filter, lgrid = 32, lgrid = 64, 
            lgrid=128,... is recommended. lgrid should be specified as {32},
            {64}, {128},... in order to easily find its existance.
    It is assumed that if wtx, type, and lgrid or two out of them are in use, 
    then they are specified in the above order.
    --------------------------------------------------------------------------
    Compared with the PMR code, there are two differences in the input data:
    First, in the PMR code, lgrid is specified inside the code and, secondly,
    in the present code, the desired function, instead of being a constant as
    in the PMR code, is given at both band edges meaning that the desired 
    function is a straight line passing two x-y coordinates. This option has 
    been introduced after following the trials of some MATLAB people to 
    translate the PMR code into MATLAB. Perhaps, those MATLAB people could not
    understand that for differentiators the desired function is a straight 
    line. Anyway, this option is now in use!
    ------------------------------------------------------------------------- 
    		Example of a lowpass filter of orddeer 30, i.e., length 31
            Err] =  EQR_FIR(30,[0 0.2 0.4 1]*pi,[1 1 0 0],[1 10], 'm');
    	
    		Example of a   differentiator
            dd, Err] = REMEZ_FIR(31,[0 1]*pi,[1 1],  'd');
    		[heven, Err] = REMEZ_FIR(30,[0 0.9*pi],[1 1], 'd');
    	
    		Example of a Hilbert filter
    		[h, Err] = REMEZ_FIR(18, [0.1 0.9]*pi, [1 1], 'h');
    ------------------------------------------------------------------------- 
    	
    Outputs
        h - coefficients of the filter
        err - the final np.absolute value of the weighted error function
    	
    References:
    [1] J. H. McClellan, T. W. Parks, and L. R. Rabiner, "A computer program 
        for designing optimum FIR linear phase digital filters," IEEE 
        Transactions on Audio and Electroacoustics, vol. AU-21, no. 6, pp.
        506 - 526, December 1973; 
     -  also reprinted in IEEE Acoustics, Speech, and Signal Processing 
        Society. Digital Signal Processing Committee, eds., "Selected 
        Papers in Digital Signal Processing, II", pp. 97 - 117, IEEE Press, 
        New York, 1976; 
     -  the FORTRAN program itself and its short description is included in 
        IEEE Acoustics, Speech, and Signal Processing  Society. Digital 
        Signal Processing Committee, eds., "Programs for Digital Signal 
        Processing", pp. 5.1-1 - 5.1-13, IEEE Press, John Wiley & Sons 
        New York, 1979.
    [2] M. Ahsan and T. Saramäki, "A MATLAB based optimum multiband FIR  
         filters design program following the original idea of the Remez 
         multiple exchange algorithm," in Proc. 2011 IEEE International 
         Symposium on Circuits and Systems, Rio de Janeiro, Brazil, 
         May 1517, 2011, pp. 137  140. 
    	
    """
    
    edge = np.array(edge) / (fs/2)


    if not isinstance(wtx , (type(None), np.ndarray, list)):
        wtx = np.ones(len(fx) // 2)
    else:
        if not isinstance(wtx , list):
            wtx = np.array(wtx)
            
    # lgrid = 16  # default value

    # ==========================================================================
    # Check varargin according to the above assumptions
    # ==========================================================================    
    # nn = len(args)
    # if nn > 0 and isinstance(args[-1], list):
    #     lgrid = args[-1][0]
    #     nn -= 1
    #     args = args[:nn]

    # if nn == 1:
    #     if isinstance(args[0], str):
    #         filter_type = args[0]
    #         wtx = np.ones(len(fx) // 2)
    #     else:
    #         wtx = np.array(args[0])
    #         filter_type = 'multiband'
    # elif nn == 2:
    #     wtx = np.array(args[0])
    #     filter_type = args[1]
    # else:
    #     wtx = np.ones(len(fx) // 2)

	#==========================================================================
	#  Find out jtype that was used in the PM code.
	#  This not necessary but simplifies the undertanding of this code snippet.
	#==========================================================================
    if filter_type.lower().startswith('d'):
        jtype = 2  # Differentiator
    elif filter_type.lower().startswith('h'):
        jtype = 3  # Hilbert transformer
    elif filter_type.lower().startswith('m'):
        jtype = 1  # Multiband filter
    else:
        jtype = 1  # Multiband filter

	#==========================================================================
	# Determine the filter cases and nfcns, the number of basis functions to be 
	# used in the Remez algorithm 
	# In the below, filtercase=1,2,3,4 is used for making it easier to 
	# understand this code snippet.   
	#==========================================================================
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

	#  ===>
	# nfcns = order/2+1 for filtercase = 1    	no fixed zeros
	# nfcns = (order+1)/2 for filtercase = 2; 	fixed zero at z = -1 ==> 
	#                                     		fixed term (1+z^{-1})/2
	# nfcns = order/2 for filtercase = 3;     	fixed zeros at z = 1 and z = -1 ==> 
	#                                     		fixed term (1-z^{-2})/2
	# nfcns = (order+1) for filtercase = 4;   	fixed zero at z = -1 ==> 
	#                                    		fixed term (1-z^{-2})/2
	#=========================================================================
	# DETERMINE grid, des, and wt 
	#========================================================================
	# Compared with the PM code, there are the following key differences:
	# (1) The upper edge for each band under consideration is automatically 
	#     included in grid. This somehow increases the accuracy. 
	# (2) Since the frequency range is now from 0 to 1, delf has been increased
	#     by a factor of 2.
	# (3) The change of des and wt depending on the filter type is peformed 
	#     before using the (modified) Remez algorithm.
	# (4) The removal of problematic angular frequencies at 0 and pi is 
	#     performed simultaneously for all filter types. Now the remomal is
	#     is performed while generating grid.
	#=========================================================================
    
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
            if np.abs(fx[2 * ll]) < 1.0e-3:
                wt_more = wtx[ll] * np.ones(number_grid + 1)
            else:
                wt_more = wtx[ll] / (grid_more * np.pi)

        des.extend(des_more)
        wt.extend(wt_more)

    grid = np.array(grid)
    des = np.array(des)
    wt = np.array(wt)

    # MLLS
    _, edge_idx, _ = np.intersect1d(grid, edge, assume_unique=True, return_indices=True)
    
    if( grid[edge_idx[-1]] != edge[-1]):
        
        edge_idx = np.append(edge_idx, [len(grid)-1])
        

	#========================================================================
	# Compared the PM code, there are two basic differences:
	# (1) In the allocation of grid points in each band, delf_new has been 
	#     predetermined such that the last point automatically occurs at the 
	#     upper edge. In the original code, it becomes, due to the use of delf, 
	#     in most cases larger than the edge. As a consequence, the last point
	#     is removed and the value of previous grid point is fixed to take on
	#     the edge value. Hence, the distance within the last and second last
	#     grid points is in most cases larger than delf.
	# (2) In the jtype=2 case, des=pi*grid*fx, instead of des=grid*fx, and
	#     wt=wtx/(pi*grid), instead of wt=wtx/grid, are used provided that 
	#     fx >= 1.0e-3.
	#==========================================================================
	# Modify des and wt depending on the filter case
	#========================================================================== 
    # if filtercase == 2:
    #     des /= np.cos(np.pi * grid / 2)
    #     wt *= np.cos(np.pi * grid / 2)
    # if filtercase == 4:
    #     des /= np.sin(np.pi * grid / 2)
    #     wt *= np.sin(np.pi * grid / 2)
    # if filtercase == 3:
    #     des /= np.sin(np.pi * grid)
    #     wt *= np.sin(np.pi * grid)

    #==========================================================================
	# CALL THE REMEZ ALGORITHM 
	#==========================================================================
	# Compared with the PM code, there are the following key differences:
	# (1) On purpose, only filters with even order and even symmetry are under
	#     consideration or filtercase = 1 is in use. As a matter of fact, the
	#     original FORTAN code did it indirectly.
	# (2) For achieving this goal, des and wt modified beforehand and the 
	#     possible response for another filter type will be generated later 
	#     based on the present response
	# (3) The Remez multiple exchange loop has been significantly improved.
	# (4) grid, the frequency grid, is now within 0 and 1, instead of being
	#     within 0 and 0.5.
	#==========================================================================
    # h, err, iext = REMEZ_EX_A(nfcns, grid, des, wt, bases_por_banda, edge, maxiter = maxit)
    h, err, iext = REMEZ_EX_MLLS(nfcns, grid, des, wt, edge_idx, maxiter = maxit, bDebug=bDebug)

    err = np.abs(err)
    return h, err, iext


# Función para filtrar los extremos consecutivos de mismo signo y mantener el de mayor módulo absoluto
def filter_extremes(Ew, peaks):
    filtered_peaks = []
    current_sign = np.sign(Ew[peaks[0]])
    max_peak = peaks[0]
    
    for peak in peaks[1:]:
        peak_sign = np.sign(Ew[peak])
        
        # Si el signo del siguiente extremo es el mismo, conservamos el de mayor módulo absoluto
        if peak_sign == current_sign:
            if np.abs(Ew[peak]) > np.abs(Ew[max_peak]):
                max_peak = peak  # Actualizamos el pico con el mayor valor absoluto
        else:
            filtered_peaks.append(max_peak)  # Guardamos el pico de mayor valor absoluto del grupo
            max_peak = peak  # Empezamos a comparar en el nuevo grupo
            current_sign = peak_sign
    
    # Agregar el último extremo
    filtered_peaks.append(max_peak)
    
    return np.array(filtered_peaks)


def REMEZ_FIR(order, edge, fx, lgrid = 16, fs = 2.0, 
              filter_type = 'multiband', wtx = None, maxiter = 250):
    """
    RERMEZ_FIR - A translation of the FORTRAN code of the Parks-McClellan
    minimax arbitrary-magnitude FIR filter design algorithm [1], referred 
    in later use to as the PMR code, into MATLAB (by Tapio Saramaki, 
    2018-02-20), and then into Python by MLLS.
    	
    Compared with the PMR code, the the peculiarities of the present code are: 
    (1) The Remez loop has been improved according to [2].
    (2) Due to the use of MATLAB, the grid pont allocation happened to be 
        improved.
    (3) The desired function is given in a slightly different manner.
    	
    Inputs to be always used:
    order - filter order
    edge  - specifies the upper and lower edges of the bands under consideration.
            The program, however, uses band edes in terms of fractions of pi rad.
    	    edge = edge/pi;
    fx -    specifies the desired values at the edges of each band.
    	
    Inputs to a possible inclusion in varargin: 
    wtx   - a constant weighting in each band
    type  - three filter types are under consideration, namely,
            (1) Filters with symmetric impulse response for multiband design: 
                lowpass, higpass, bandpass, and bandstop designs are typical
                examples. The benefit of these filters is that the phase is
                perfectly linear for all frequencies.
            (2) Full band and partial band differentiators: filters with 
                antisymmetric impulse response are in use.
            (3) Hilbert transformers: again antisymmetry is required.
            The distinction between these three types is performed by a 
            character array, called type, such that
            type(1)='m' or type(1)='M' ==> 'multiband' design
            type(1)='d' or type(1)='D' ==> 'differentiator'
            type(1)='h' or type(1)='H' ==> 'Hilbert' tranformer
            If type is not specified, then type=='multiband' is used
    lgrid - if not specified, lgrid=16 as in the PMR code. To  increase the 
            accuray of the resulting filter, lgrid = 32, lgrid = 64, 
            lgrid=128,... is recommended. lgrid should be specified as {32},
            {64}, {128},... in order to easily find its existance.
    It is assumed that if wtx, type, and lgrid or two out of them are in use, 
    then they are specified in the above order.
    --------------------------------------------------------------------------
    Compared with the PMR code, there are two differences in the input data:
    First, in the PMR code, lgrid is specified inside the code and, secondly,
    in the present code, the desired function, instead of being a constant as
    in the PMR code, is given at both band edges meaning that the desired 
    function is a straight line passing two x-y coordinates. This option has 
    been introduced after following the trials of some MATLAB people to 
    translate the PMR code into MATLAB. Perhaps, those MATLAB people could not
    understand that for differentiators the desired function is a straight 
    line. Anyway, this option is now in use!
    ------------------------------------------------------------------------- 
    		Example of a lowpass filter of orddeer 30, i.e., length 31
            Err] =  EQR_FIR(30,[0 0.2 0.4 1]*pi,[1 1 0 0],[1 10], 'm');
    	
    		Example of a   differentiator
            dd, Err] = REMEZ_FIR(31,[0 1]*pi,[1 1],  'd');
    		[heven, Err] = REMEZ_FIR(30,[0 0.9*pi],[1 1], 'd');
    	
    		Example of a Hilbert filter
    		[h, Err] = REMEZ_FIR(18, [0.1 0.9]*pi, [1 1], 'h');
    ------------------------------------------------------------------------- 
    	
    Outputs
        h - coefficients of the filter
        err - the final np.absolute value of the weighted error function
    	
    References:
    [1] J. H. McClellan, T. W. Parks, and L. R. Rabiner, "A computer program 
        for designing optimum FIR linear phase digital filters," IEEE 
        Transactions on Audio and Electroacoustics, vol. AU-21, no. 6, pp.
        506 - 526, December 1973; 
     -  also reprinted in IEEE Acoustics, Speech, and Signal Processing 
        Society. Digital Signal Processing Committee, eds., "Selected 
        Papers in Digital Signal Processing, II", pp. 97 - 117, IEEE Press, 
        New York, 1976; 
     -  the FORTRAN program itself and its short description is included in 
        IEEE Acoustics, Speech, and Signal Processing  Society. Digital 
        Signal Processing Committee, eds., "Programs for Digital Signal 
        Processing", pp. 5.1-1 - 5.1-13, IEEE Press, John Wiley & Sons 
        New York, 1979.
    [2] M. Ahsan and T. Saramäki, "A MATLAB based optimum multiband FIR  
         filters design program following the original idea of the Remez 
         multiple exchange algorithm," in Proc. 2011 IEEE International 
         Symposium on Circuits and Systems, Rio de Janeiro, Brazil, 
         May 1517, 2011, pp. 137  140. 
    	
    """
    
    edge = np.array(edge) / (fs/2)


    if not isinstance(wtx , (type(None), np.ndarray, list)):
        wtx = np.ones(len(fx) // 2)
    else:
        if not isinstance(wtx , list):
            wtx = np.array(wtx)
            
    # lgrid = 16  # default value

    # ==========================================================================
    # Check varargin according to the above assumptions
    # ==========================================================================    
    # nn = len(args)
    # if nn > 0 and isinstance(args[-1], list):
    #     lgrid = args[-1][0]
    #     nn -= 1
    #     args = args[:nn]

    # if nn == 1:
    #     if isinstance(args[0], str):
    #         filter_type = args[0]
    #         wtx = np.ones(len(fx) // 2)
    #     else:
    #         wtx = np.array(args[0])
    #         filter_type = 'multiband'
    # elif nn == 2:
    #     wtx = np.array(args[0])
    #     filter_type = args[1]
    # else:
    #     wtx = np.ones(len(fx) // 2)

	#==========================================================================
	#  Find out jtype that was used in the PM code.
	#  This not necessary but simplifies the undertanding of this code snippet.
	#==========================================================================
    if filter_type.lower().startswith('d'):
        jtype = 2  # Differentiator
    elif filter_type.lower().startswith('h'):
        jtype = 3  # Hilbert transformer
    elif filter_type.lower().startswith('m'):
        jtype = 1  # Multiband filter
    else:
        jtype = 1  # Multiband filter

	#==========================================================================
	# Determine the filter cases and nfcns, the number of basis functions to be 
	# used in the Remez algorithm 
	# In the below, filtercase=1,2,3,4 is used for making it easier to 
	# understand this code snippet.   
	#==========================================================================
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

	#  ===>
	#nfcns = order/2+1 for filtercase = 1    	no fixed zeros
	#nfcns = (order+1)/2 for filtercase = 2; 	fixed zero at z = -1 ==> 
	#                                     		fixed term (1+z^{-1})/2
	#nfcns = order/2 for filtercase = 3;     	fixed zeros at z = 1 and z = -1 ==> 
	#                                     		fixed term (1-z^{-2})/2
	#nfcns = (order+1) for filtercase = 4;   	fixed zero at z = -1 ==> 
	#                                    		fixed term (1-z^{-2})/2
	#=========================================================================
	# DETERMINE grid, des, and wt 
	#========================================================================
	# Compared with the PM code, there are the following key differences:
	# (1) The upper edge for each band under consideration is automatically 
	#     included in grid. This somehow increases the accuracy. 
	# (2) Since the frequency range is now from 0 to 1, delf has been increased
	#     by a factor of 2.
	# (3) The change of des and wt depending on the filter type is peformed 
	#     before using the (modified) Remez algorithm.
	# (4) The removal of problematic angular frequencies at 0 and pi is 
	#     performed simultaneously for all filter types. Now the remomal is
	#     is performed while generating grid.
	#=========================================================================
    
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
            if np.abs(fx[2 * ll]) < 1.0e-3:
                wt_more = wtx[ll] * np.ones(number_grid + 1)
            else:
                wt_more = wtx[ll] / (grid_more * np.pi)

        des.extend(des_more)
        wt.extend(wt_more)

    grid = np.array(grid)
    des = np.array(des)
    wt = np.array(wt)

	#========================================================================
	# Compared the PM code, there are two basic differences:
	# (1) In the allocation of grid points in each band, delf_new has been 
	#     predetermined such that the last point automatically occurs at the 
	#     upper edge. In the original code, it becomes, due to the use of delf, 
	#     in most cases larger than the edge. As a consequence, the last point
	#     is removed and the value of previous grid point is fixed to take on
	#     the edge value. Hence, the distance within the last and second last
	#     grid points is in most cases larger than delf.
	# (2) In the jtype=2 case, des=pi*grid*fx, instead of des=grid*fx, and
	#     wt=wtx/(pi*grid), instead of wt=wtx/grid, are used provided that 
	#     fx >= 1.0e-3.
	#==========================================================================
	# Modify des and wt depending on the filter case
	#========================================================================== 
    if filtercase == 2:
        h = np.concatenate(([h[0] / 2], (h[1:] + h[:-1]) / 2, [h[-1] / 2]))
    if filtercase == 3:
        h = np.concatenate(([h[0] / 2, h[1] / 2], (h[2:] - h[:-2]) / 2, [-h[-2] / 2, -h[-1] / 2]))
    if filtercase == 4:
        h = np.concatenate(([h[0] / 2], (h[1:] - h[:-1]) / 2, [-h[-1] / 2]))

    err = np.abs(err)
    return h, err, iext


# Función para filtrar los extremos consecutivos de mismo signo y mantener el de mayor módulo absoluto
def filter_extremes(Ew, peaks):
    filtered_peaks = []
    current_sign = np.sign(Ew[peaks[0]])
    max_peak = peaks[0]
    
    for peak in peaks[1:]:
        peak_sign = np.sign(Ew[peak])
        
        # Si el signo del siguiente extremo es el mismo, conservamos el de mayor módulo absoluto
        if peak_sign == current_sign:
            if np.abs(Ew[peak]) > np.abs(Ew[max_peak]):
                max_peak = peak  # Actualizamos el pico con el mayor valor absoluto
        else:
            filtered_peaks.append(max_peak)  # Guardamos el pico de mayor valor absoluto del grupo
            max_peak = peak  # Empezamos a comparar en el nuevo grupo
            current_sign = peak_sign
    
    # Agregar el último extremo
    filtered_peaks.append(max_peak)
    
    return np.array(filtered_peaks)


def REMEZ_FIR(order, edge, fx, lgrid = 16, fs = 2.0, 
              filter_type = 'multiband', wtx = None, maxiter = 250):
    """
    RERMEZ_FIR - A translation of the FORTRAN code of the Parks-McClellan
    minimax arbitrary-magnitude FIR filter design algorithm [1], referred 
    in later use to as the PMR code, into MATLAB (by Tapio Saramaki, 
    2018-02-20), and then into Python by MLLS.
    	
    Compared with the PMR code, the the peculiarities of the present code are: 
    (1) The Remez loop has been improved according to [2].
    (2) Due to the use of MATLAB, the grid pont allocation happened to be 
        improved.
    (3) The desired function is given in a slightly different manner.
    	
    Inputs to be always used:
    order - filter order
    edge  - specifies the upper and lower edges of the bands under consideration.
            The program, however, uses band edes in terms of fractions of pi rad.
    	    edge = edge/pi;
    fx -    specifies the desired values at the edges of each band.
    	
    Inputs to a possible inclusion in varargin: 
    wtx   - a constant weighting in each band
    type  - three filter types are under consideration, namely,
            (1) Filters with symmetric impulse response for multiband design: 
                lowpass, higpass, bandpass, and bandstop designs are typical
                examples. The benefit of these filters is that the phase is
                perfectly linear for all frequencies.
            (2) Full band and partial band differentiators: filters with 
                antisymmetric impulse response are in use.
            (3) Hilbert transformers: again antisymmetry is required.
            The distinction between these three types is performed by a 
            character array, called type, such that
            type(1)='m' or type(1)='M' ==> 'multiband' design
            type(1)='d' or type(1)='D' ==> 'differentiator'
            type(1)='h' or type(1)='H' ==> 'Hilbert' tranformer
            If type is not specified, then type=='multiband' is used
    lgrid - if not specified, lgrid=16 as in the PMR code. To  increase the 
            accuray of the resulting filter, lgrid = 32, lgrid = 64, 
            lgrid=128,... is recommended. lgrid should be specified as {32},
            {64}, {128},... in order to easily find its existance.
    It is assumed that if wtx, type, and lgrid or two out of them are in use, 
    then they are specified in the above order.
    --------------------------------------------------------------------------
    Compared with the PMR code, there are two differences in the input data:
    First, in the PMR code, lgrid is specified inside the code and, secondly,
    in the present code, the desired function, instead of being a constant as
    in the PMR code, is given at both band edges meaning that the desired 
    function is a straight line passing two x-y coordinates. This option has 
    been introduced after following the trials of some MATLAB people to 
    translate the PMR code into MATLAB. Perhaps, those MATLAB people could not
    understand that for differentiators the desired function is a straight 
    line. Anyway, this option is now in use!
    ------------------------------------------------------------------------- 
    		Example of a lowpass filter of orddeer 30, i.e., length 31
            Err] =  EQR_FIR(30,[0 0.2 0.4 1]*pi,[1 1 0 0],[1 10], 'm');
    	
    		Example of a   differentiator
            dd, Err] = REMEZ_FIR(31,[0 1]*pi,[1 1],  'd');
    		[heven, Err] = REMEZ_FIR(30,[0 0.9*pi],[1 1], 'd');
    	
    		Example of a Hilbert filter
    		[h, Err] = REMEZ_FIR(18, [0.1 0.9]*pi, [1 1], 'h');
    ------------------------------------------------------------------------- 
    	
    Outputs
        h - coefficients of the filter
        err - the final np.absolute value of the weighted error function
    	
    References:
    [1] J. H. McClellan, T. W. Parks, and L. R. Rabiner, "A computer program 
        for designing optimum FIR linear phase digital filters," IEEE 
        Transactions on Audio and Electroacoustics, vol. AU-21, no. 6, pp.
        506 - 526, December 1973; 
     -  also reprinted in IEEE Acoustics, Speech, and Signal Processing 
        Society. Digital Signal Processing Committee, eds., "Selected 
        Papers in Digital Signal Processing, II", pp. 97 - 117, IEEE Press, 
        New York, 1976; 
     -  the FORTRAN program itself and its short description is included in 
        IEEE Acoustics, Speech, and Signal Processing  Society. Digital 
        Signal Processing Committee, eds., "Programs for Digital Signal 
        Processing", pp. 5.1-1 - 5.1-13, IEEE Press, John Wiley & Sons 
        New York, 1979.
    [2] M. Ahsan and T. Saramäki, "A MATLAB based optimum multiband FIR  
         filters design program following the original idea of the Remez 
         multiple exchange algorithm," in Proc. 2011 IEEE International 
         Symposium on Circuits and Systems, Rio de Janeiro, Brazil, 
         May 1517, 2011, pp. 137  140. 
    	
    """
    
    edge = np.array(edge) / (fs/2)


    if not isinstance(wtx , (type(None), np.ndarray, list)):
        wtx = np.ones(len(fx) // 2)
    else:
        if not isinstance(wtx , list):
            wtx = np.array(wtx)
            
    # lgrid = 16  # default value

    # ==========================================================================
    # Check varargin according to the above assumptions
    # ==========================================================================    
    # nn = len(args)
    # if nn > 0 and isinstance(args[-1], list):
    #     lgrid = args[-1][0]
    #     nn -= 1
    #     args = args[:nn]

    # if nn == 1:
    #     if isinstance(args[0], str):
    #         filter_type = args[0]
    #         wtx = np.ones(len(fx) // 2)
    #     else:
    #         wtx = np.array(args[0])
    #         filter_type = 'multiband'
    # elif nn == 2:
    #     wtx = np.array(args[0])
    #     filter_type = args[1]
    # else:
    #     wtx = np.ones(len(fx) // 2)

	#==========================================================================
	#  Find out jtype that was used in the PM code.
	#  This not necessary but simplifies the undertanding of this code snippet.
	#==========================================================================
    if filter_type.lower().startswith('d'):
        jtype = 2  # Differentiator
    elif filter_type.lower().startswith('h'):
        jtype = 3  # Hilbert transformer
    elif filter_type.lower().startswith('m'):
        jtype = 1  # Multiband filter
    else:
        jtype = 1  # Multiband filter

	#==========================================================================
	# Determine the filter cases and nfcns, the number of basis functions to be 
	# used in the Remez algorithm 
	# In the below, filtercase=1,2,3,4 is used for making it easier to 
	# understand this code snippet.   
	#==========================================================================
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

	#  ===>
	#nfcns = order/2+1 for filtercase = 1    	no fixed zeros
	#nfcns = (order+1)/2 for filtercase = 2; 	fixed zero at z = -1 ==> 
	#                                     		fixed term (1+z^{-1})/2
	#nfcns = order/2 for filtercase = 3;     	fixed zeros at z = 1 and z = -1 ==> 
	#                                     		fixed term (1-z^{-2})/2
	#nfcns = (order+1) for filtercase = 4;   	fixed zero at z = -1 ==> 
	#                                    		fixed term (1-z^{-2})/2
	#=========================================================================
	# DETERMINE grid, des, and wt 
	#========================================================================
	# Compared with the PM code, there are the following key differences:
	# (1) The upper edge for each band under consideration is automatically 
	#     included in grid. This somehow increases the accuracy. 
	# (2) Since the frequency range is now from 0 to 1, delf has been increased
	#     by a factor of 2.
	# (3) The change of des and wt depending on the filter type is peformed 
	#     before using the (modified) Remez algorithm.
	# (4) The removal of problematic angular frequencies at 0 and pi is 
	#     performed simultaneously for all filter types. Now the remomal is
	#     is performed while generating grid.
	#=========================================================================
    
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
            if np.abs(fx[2 * ll]) < 1.0e-3:
                wt_more = wtx[ll] * np.ones(number_grid + 1)
            else:
                wt_more = wtx[ll] / (grid_more * np.pi)

        des.extend(des_more)
        wt.extend(wt_more)

    grid = np.array(grid)
    des = np.array(des)
    wt = np.array(wt)

	#========================================================================
	# Compared the PM code, there are two basic differences:
	# (1) In the allocation of grid points in each band, delf_new has been 
	#     predetermined such that the last point automatically occurs at the 
	#     upper edge. In the original code, it becomes, due to the use of delf, 
	#     in most cases larger than the edge. As a consequence, the last point
	#     is removed and the value of previous grid point is fixed to take on
	#     the edge value. Hence, the distance within the last and second last
	#     grid points is in most cases larger than delf.
	# (2) In the jtype=2 case, des=pi*grid*fx, instead of des=grid*fx, and
	#     wt=wtx/(pi*grid), instead of wt=wtx/grid, are used provided that 
	#     fx >= 1.0e-3.
	#==========================================================================
	# Modify des and wt depending on the filter case
	#========================================================================== 
    if filtercase == 2:
        des /= np.cos(np.pi * grid / 2)
        wt *= np.cos(np.pi * grid / 2)
    if filtercase == 4:
        des /= np.sin(np.pi * grid / 2)
        wt *= np.sin(np.pi * grid / 2)
    if filtercase == 3:
        des /= np.sin(np.pi * grid)
        wt *= np.sin(np.pi * grid)

    #==========================================================================
	# CALL THE REMEZ ALGORITHM 
	#==========================================================================
	# Compared with the PM code, there are the following key differences:
	# (1) On purpose, only filters with even order and even symmetry are under
	#     consideration or filtercase = 1 is in use. As a matter of fact, the
	#     original FORTAN code did it indirectly.
	# (2) For achieving this goal, des and wt modified beforehand and the 
	#     possible response for another filter type will be generated later 
	#     based on the present response
	# (3) The Remez multiple exchange loop has been significantly improved.
	# (4) grid, the frequency grid, is now within 0 and 1, instead of being
	#     within 0 and 0.5.
	#==========================================================================
    h, err, iext = REMEZ_EX_A(nfcns, grid, des, wt, edge, maxiter = maxit)

    # Generate the impulse responses for other types
    nn = len(h)
    if filtercase == 2:
        h = np.concatenate(([h[0] / 2], (h[1:] + h[:-1]) / 2, [h[-1] / 2]))
    if filtercase == 3:
        h = np.concatenate(([h[0] / 2, h[1] / 2], (h[2:] - h[:-2]) / 2, [-h[-2] / 2, -h[-1] / 2]))
    if filtercase == 4:
        h = np.concatenate(([h[0] / 2], (h[1:] - h[:-1]) / 2, [-h[-1] / 2]))

    err = np.abs(err)
    return h, err, iext


def REMEZ_EX_MLLS(M, grid, des, wt, edge_idx, maxiter = 250, error_tol = 10e-4, bDebug = False):
    
    # Initializations
    ngrid = len(grid)
    # l_ove = np.arange(ngrid)

    # Definir frecuencias extremas iniciales
    omega_scale = (ngrid - 1) / M
    jj = np.arange(M)
    omega_ext_iniciales_idx = np.concatenate((np.fix(omega_scale * jj), [ngrid-1])).astype(int)

    
    # aseguro que siempre haya una omega extrema en los edges.
    aux_idx = np.array([np.argmin(np.abs(grid[omega_ext_iniciales_idx] - grid[ii])) for ii in edge_idx])
    omega_ext_iniciales_idx[aux_idx] = edge_idx

    ## Debug

    fs = 2.0
    fft_sz = 512
    half_fft_sz = fft_sz//2
    frecuencias = np.arange(start=0, stop=fs, step=fs/fft_sz )

    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()

    if bDebug:
        ## Debug
        plt.figure(1)
        plt.clf()
        plt.figure(2)
        plt.clf()
        plt.figure(3)
        plt.clf()
        D_ext = np.interp(frecuencias[:half_fft_sz], grid, des)
        plt.plot(frecuencias[:half_fft_sz], D_ext, label='D($\Omega$)')
        ## Debug
    
    niter = 1

    omega_ext_idx = omega_ext_iniciales_idx
    omega_ext_prev_idx = np.zeros_like(omega_ext_idx)

    prev_error_target = np.finfo(np.float64).max
    
    # Remez loop
    while niter < maxiter:

        # Construir el sistema de ecuaciones a partir de la matriz de diseño A.
        A = np.zeros((M+1, M+1))
        for ii, omega_idx in enumerate(omega_ext_idx):
            A[ii,:] = np.hstack((np.cos( np.pi * grid[omega_idx] * np.arange(M)), (-1)**ii/wt[omega_idx]))

        # Resolver el sistema de ecuaciones para los coeficientes únicos
        xx = np.linalg.solve(A, des[omega_ext_idx])
        
        # los primeros resultados están realacionados a los coeficientes del filtro
        a_coeffs_half = xx[:-1]
        # el último es el error cometido en la aproximación
        this_error_target = np.abs(xx[-1])

        # Construimos la respuesta interpolada en "grid" para refinar las 
        # frecuencias extremas
        Aw_grid = np.zeros(ngrid)
        for ii in range(M):
            Aw_grid  += a_coeffs_half[ii] * np.cos( ii * np.pi * grid )

        # Calculamos la secuencia de error pesado: nos van a interesar los 
        # signos en las omega extremas para filtrar aquellas omega que NO 
        # alternan.
        Ew = wt*(des - Aw_grid)
        # también el módulo para verificar que ninguno esté por encima del 
        # error cometido "this_error_target"
        Ew_abs = np.abs(Ew)
        
        # procedemos a filtrar las omega extremas.
        peaks_pos , _ = find_peaks(Ew, height= 0.0)
        peaks_neg , _ = find_peaks(-Ew, height= 0.0)
        peaks = np.sort(np.concatenate((peaks_pos,peaks_neg)))
        
        # Aplicar el filtro a los picos encontrados
        peaks = filter_extremes(Ew, peaks)

        omega_ext_idx = np.unique(np.concatenate((edge_idx, peaks)))

        omega_ext_idx = filter_extremes(Ew, omega_ext_idx)

        # probamos si converge exitosamente
        if np.all(Ew_abs[omega_ext_idx] - this_error_target < error_tol):
            
            print("Convergencia exitosa!")
            break
        
        # Problemas en la convergencia: sin cambios en el error ni las frecuencias extremas 
        elif this_error_target  == prev_error_target and np.array_equal(omega_ext_idx, omega_ext_prev_idx):
            
            print("Problemas de convergencia:")
            break

        if bDebug:
            ## Debug
            # Graficar la respuesta en frecuencia
            plt.figure(1)
            # plt.clf()
            # plt.plot(frecuencias[:half_fft_sz], Aw_ext, label=f'Aw_ext {niter}')
            # plt.plot(grid[omega_ext_idx], Aw, 'ob')
            # plt.plot(frecuencias[:half_fft_sz], W_err_orig, label=f'orig {niter}')
        
            # plt.plot(grid, Ew, label=f'$E_{niter}$')
            plt.plot(grid, Ew)
            plt.plot(grid[omega_ext_prev_idx], Ew[omega_ext_prev_idx], 'or')
            # plt.plot(frecuencias[:half_fft_sz], w_err_ext, label=f'Ew_ext {niter}')
            plt.plot(grid[omega_ext_idx], Ew[omega_ext_idx], 'xb')
            plt.plot([ 0, 1], [0, 0], '-k', lw=0.8)
            plt.plot([ 0, 1], [this_error_target, this_error_target], ':k', lw=0.8, label=f'$\delta_{niter}=$ {this_error_target:3.3f}')
            plt.plot([ 0, 1], [-this_error_target, -this_error_target], ':k', lw=0.8)
        
            plt.title("Error pesado: $E(\Omega) = W(\Omega) \cdot [D(\Omega) - H_R(\Omega)]$")
            plt.xlabel("Frecuencia Normalizada")
            plt.ylabel("Magnitud")
            plt.legend()
        
            a_coeffs_half = xx[:-1]
            a_coeffs_half[1:] = a_coeffs_half[1:]/2
            h_coeffs = np.concatenate((a_coeffs_half[::-1], a_coeffs_half[1:]))
        
            H = np.fft.fft(h_coeffs, fft_sz)
        
            plt.figure(2)
            plt.plot(frecuencias[:half_fft_sz], 20*np.log10(np.abs(H[:half_fft_sz])), label=f'Iter: {niter}')
    
            plt.title("Respuesta en frecuencia de módulo: $ \\left|H(\Omega)\\right| $")
            plt.xlabel("Frecuencia Normalizada")
            plt.ylabel("$\\left|H(\Omega)\\right|_{{dB}}$")
            plt.legend()
        
            plt.figure(3)
            Aw_ext = np.interp(frecuencias[:half_fft_sz], grid, Aw_grid)
            plt.plot(frecuencias[:half_fft_sz], Aw_ext, label=f'$H_{{R{niter}}}$')
            plt.legend()
            plt.show()
            pass
    
            ## Debug

        # continuamos buscando la convergencia
        omega_ext_prev_idx = omega_ext_idx
        prev_error_target = this_error_target
        niter += 1


    # coeficientes del filtro        
    a_coeffs_half = xx[:-1]
    a_coeffs_half[1:] = a_coeffs_half[1:]/2

    h_coeffs = np.concatenate((a_coeffs_half[::-1], a_coeffs_half[1:]))

    ## Debug
    if bDebug:
        # Graficar la respuesta en frecuencia
        plt.figure(1)
        # plt.clf()
        # plt.plot(frecuencias[:half_fft_sz], Aw_ext, label=f'Aw_ext {niter}')
        # plt.plot(grid[omega_ext_idx], Aw, 'ob')
        # plt.plot(frecuencias[:half_fft_sz], W_err_orig, label=f'orig {niter}')
    
        # plt.plot(grid, Ew, label=f'$E_{niter}$')
        plt.plot(grid, Ew)
        plt.plot(grid[omega_ext_prev_idx], Ew[omega_ext_prev_idx], 'or')
        # plt.plot(frecuencias[:half_fft_sz], w_err_ext, label=f'Ew_ext {niter}')
        plt.plot(grid[omega_ext_idx], Ew[omega_ext_idx], 'xb')
        plt.plot([ 0, 1], [0, 0], '-k', lw=0.8)
        plt.plot([ 0, 1], [this_error_target, this_error_target], ':k', lw=0.8, label=f'$\delta_{niter}=$ {this_error_target:3.3f}')
        plt.plot([ 0, 1], [-this_error_target, -this_error_target], ':k', lw=0.8)
    
        plt.title("Error pesado: $E(\Omega) = W(\Omega) \cdot [D(\Omega) - H_R(\Omega)]$")
        plt.xlabel("Frecuencia Normalizada")
        plt.ylabel("Magnitud")
        plt.legend()
    
        H = np.fft.fft(h_coeffs, fft_sz)
    
        plt.figure(2)
        plt.plot(frecuencias[:half_fft_sz], 20*np.log10(np.abs(H[:half_fft_sz])), label=f'Iter: {niter}')

        plt.title("Respuesta en frecuencia de módulo: $ \\left|H(\Omega)\\right| $")
        plt.xlabel("Frecuencia Normalizada")
        plt.ylabel("$\\left|H(\Omega)\\right|_{{dB}}$")
        plt.legend()
    
        plt.figure(3)
        Aw_ext = np.interp(frecuencias[:half_fft_sz], grid, Aw_grid)
        plt.plot(frecuencias[:half_fft_sz], Aw_ext, label=f'$H_{{R{niter}}}$')
        plt.legend()
        plt.show()
    
        ## Debug

    return h_coeffs, this_error_target, grid[omega_ext_idx]


def REMEZ_EX_A(nfcns, grid, des, wt, edges, maxiter = 250):
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
    
	 Author: 		Tapio Saramaki, 2018-01-20    
    """
    
    # Initializations
    ngrid = len(grid)
    l_ove = np.arange(ngrid)
    
    # (original) primer intento de hallar las Omega extremas
    temp = (ngrid - 1) / nfcns
    jj = np.arange(nfcns)
    l_trial = np.concatenate((np.fix(temp * jj), [ngrid-1])).astype(int)
    
    nz = nfcns + 1
    devl = 0
    niter = 1
    x_all = np.cos(np.pi * grid)
    jj = np.arange(nfcns)
    
    # Remez loop
    while niter < maxiter:
        
        # print(f'iter: {niter}' )
        
        x = np.cos(np.pi * grid[l_trial])
        
# 		--------------------------------------------------------------------------
# 		 The following way of calculating the Lagrange interpolation coefficients
# 		 has been copied from the PM code. The original idea is excellent because
# 		 the high accuracy is maintained up to the filter order of 2000
# 		--------------------------------------------------------------------------
        
        # Calculate the Lagrange interpolation coefficients
        jet = (nfcns - 1) // 15 + 1
        ad = np.zeros(nz)
        
        for mm in range(nz):
            yy = 1
            for nn in range(jet):
                xx = 2 * (x[mm] - x[nn:nz:jet])
                yy *= np.prod(xx[xx != 0])
            ad[mm] = 1 / yy
        
        alter = np.ones_like(ad)
        alter[1::2] = -alter[1::2]
        dnum = np.dot(ad, des[l_trial])
        dden = np.dot(alter, ad / wt[l_trial])
        dev = -dnum / dden

        # print('ad: {:3.5f}'.format(np.median(ad)))
        # print(f'dnum: {dnum}' )
        # print(f'dden: {dden}' )
        
        if np.abs(dev) <= np.abs(devl):
# 			#############################################
# 		    a need to use a more informative message.
# 			#############################################            
            warnings.warn('Convergence problems')
            break
        
        devl = dev
        y = des[l_trial] + dev * alter / wt[l_trial]
        l_left = np.setdiff1d(l_ove, l_trial)
        err_num = np.zeros_like(l_left, dtype=float)
        err_den = np.zeros_like(l_left, dtype=float)
        
        for jj in range(nz):
            aid = ad[jj] / (x_all[l_left] - x[jj])
            err_den += aid
            err_num += y[jj] * aid
            # print('aid: {:3.5f} err_num: {:3.5f} err_den: {:3.5f}'.format(np.median(aid), np.median(err_num), np.median(err_den)) )
        
        wei_err = np.zeros_like(des)
        wei_err[l_left] = (err_num / err_den - des[l_left]) * wt[l_left]
        wei_err[l_trial] = alter * dev
        # print('wei_err: {:3.5f}'.format(np.median(wei_err)))
        
        # print('wei_err[l_trial]: {:3.5f}'.format(np.median(wei_err[l_trial])))
        
        l_aid1 = np.where(np.diff(np.sign(np.diff(np.concatenate(([0], wei_err, [0]))))))[0] 
        l_aid2 = l_aid1[np.abs(wei_err[l_aid1]) >= np.abs(dev)]

        # print('wei_err[l_aid2]: {:3.5f}'.format(np.median(wei_err[l_aid2])))

        #print('wei_err: {:3.5f}'.format(np.median(wei_err)))
        
        # Indices donde cambia el signo del error ponderado
        changes = np.cumsum( np.concatenate( ([1], (np.sign(wei_err[l_aid2[1:]]) != np.sign(wei_err[l_aid2[:-1]])).astype(int)) ) ).astype(int) - 1

        sparse_mat = coo_matrix((np.abs(wei_err[l_aid2]), (np.arange( len(l_aid2) ).astype(int),  changes) ));
        
        ind = sparse_mat.argmax(axis = 0)

        l_real_init = l_aid2[ind].flatten()
        
        if (len(l_real_init) - nz) % 2 == 1:
            if np.abs(wei_err[l_real_init[0]]) <= np.abs(wei_err[l_real_init[-1]]):
                l_real_init = l_real_init[1:]
            else:
                l_real_init = l_real_init[:-1]
        
        while len(l_real_init) > nz:
            wei_real = np.abs(wei_err[l_real_init])
            wei_comp = np.maximum(wei_real[:-1], wei_real[1:])
            if np.max(np.abs(wei_err[l_real_init[0]]), np.abs(wei_err[l_real_init[-1]])) <= np.min(wei_comp):
                l_real_init = l_real_init[1:-1]
            else:
                ind_omit = np.argmin(wei_comp)
                l_real_init = np.delete(l_real_init, [ind_omit, ind_omit + 1])
        
        l_real = l_real_init

        
       
        # valor de frecuencias extremas para devolver, normalizado de 0 a 1
        iext = l_real/(ngrid-1)
        
        # ==========================================================================
        # Generate the impulse response of the filtercase = 1 filter using the IDFT.  
        # It is not very straightforward because the result of the Remez loop is
        # expressed using the Lagrange interpolation formula in barycentric form
        # The generation of this impulse response follows exactly the idea in the 
        # the PMR code.
        # ====================================================================
        
        # Generate the impulse response using the IDFT
        cn = 2 * nfcns - 1
        x_IDFT = np.arange(start=0, stop = 2 * nfcns / cn, step = 2/cn)
        x_IDFT = np.cos(np.pi * x_IDFT)
        _, ind1, ind2 = np.intersect1d(x_IDFT, x, return_indices=True)
        ind1 = np.sort(ind1)
        ind2 = np.sort(ind2)
    
        l_ove = np.arange(nfcns)
        l_left = np.setdiff1d(l_ove, ind1)
        num = np.zeros(len(l_left))
        den = np.zeros(len(l_left))
        
        for jj in range(nz):
            aid = ad[jj] / (x_IDFT[l_left] - x[jj])
            den += aid
            num += y[jj] * aid
        
        A = np.zeros(l_ove.shape)
        A[l_left] = num / den
        A[ind1] = y[ind2]
       
        if np.array_equal(l_real, l_trial):
            break
        else:
            l_trial = l_real
            niter += 1

    ## Debug

    plt.show()

    ## Debug
    
    # valor de frecuencias extremas para devolver, normalizado de 0 a 1
    iext = l_real/(ngrid-1)
    
    # ==========================================================================
    # Generate the impulse response of the filtercase = 1 filter using the IDFT.  
    # It is not very straightforward because the result of the Remez loop is
    # expressed using the Lagrange interpolation formula in barycentric form
    # The generation of this impulse response follows exactly the idea in the 
    # the PMR code.
    # ====================================================================
    
    # Generate the impulse response using the IDFT
    cn = 2 * nfcns - 1
    x_IDFT = np.arange(start=0, stop = 2 * nfcns / cn, step = 2/cn)
    x_IDFT = np.cos(np.pi * x_IDFT)
    _, ind1, ind2 = np.intersect1d(x_IDFT, x, return_indices=True)
    ind1 = np.sort(ind1)
    ind2 = np.sort(ind2)

    l_ove = np.arange(nfcns)
    l_left = np.setdiff1d(l_ove, ind1)
    num = np.zeros(len(l_left))
    den = np.zeros(len(l_left))
    
    for jj in range(nz):
        aid = ad[jj] / (x_IDFT[l_left] - x[jj])
        den += aid
        num += y[jj] * aid
    
    A = np.zeros(l_ove.shape)
    A[l_left] = num / den
    A[ind1] = y[ind2]
    
    h = np.zeros(nfcns)
    
    for n in range(1,nfcns+1):
        h[n-1] = (1 / cn) * (A[0] + 2 * np.sum(A[1:nfcns] * np.cos(2 * np.pi * np.arange(1, nfcns) * (n - 1) / cn)))
    
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


def L_PHASE_LP_FIR_ORDER(edge, d, fs = 2.0):
    """
    Estimates the required order, N, for minimax designed linear-phase 
    lowpass FIR filters.
    """
    fny = fs/2
    edge = np.array(edge) / fny
    
    if edge[0] >= edge[1] or edge[0] >= 1.0 or edge[1] > 1.0:
        raise ValueError("Improper band edges. Edges should be between 0 and 1 and in increasing order.")

    fc = edge[0] / 2   # Normalize to half the sampling frequency
    fs = edge[1] / 2
    deltaF = fs - fc
    deltac = d[0]
    deltas = d[1]
    N = N4(0.5 - fc, deltaF, deltac, deltas) - 1

    Be = np.array([0.0, edge[0], edge[1], 1.0]) * fny
    
    # orig
    D = np.array([1, 1, 0, 0])
    # MLLS
    # D = np.array([1, 1, d[1], d[1]])
    
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


from pytc2.sistemas_lineales import plot_plantilla

# Parámetros PM algorithm
fs = 2.0
maxit = 100
lgrid = 16

filter_type = 'lowpass'

fpass = 0.4 # 
ripple = 0.5 # dB
fstop = 0.6 # Hz
attenuation = 40 # dB

# Define the band edges and gains
wTedges = np.array([fpass, fstop])

b = [1, 0]  # Gains in the bands
d = np.array([-ripple, -attenuation])  # Acceptable deviations
d = 10**(d/20)
d = np.array([(1-d[0]), d[1] ])

# Compute the required stopband attenuation and passband ripple
Amaxreq = 20 * np.log10(1 + d[0] / (1 - d[0]))
Aminreq = 20 * np.log10((1 + d[0]) / d[1])

print(f"Amaxreq = {Amaxreq:.4f} dB")
print(f"Aminreq = {Aminreq:.4f} dB")

# Estimate filter order
N, Be, D, W = L_PHASE_LP_FIR_ORDER(wTedges, d)

# Output estimated order
print(f"Estimated order (N): {N}")

# Forzamos FIR tipo I
if N%2 == 0:
    N += 1  # 
print(f"Final filter order: {N}")

# Filter type (multiband)
Ftype = 'm'

# plt.close('all')

# # Design the filter using the Remez algorithm
# hh, Err, wext = REMEZ_FIR(order=N, edge=Be, fx=D, 
#                           wtx = W, filter_type = Ftype, 
#                           lgrid = lgrid, maxiter=maxit)

# # Calculate resulting passband and stopband ripples
# deltac = Err / W[0]
# deltas = Err / W[1]

# # Normalize the maximum gain to 1
# hh_tapio = hh / (1 + deltac)

# Design the filter using the Remez algorithm
hh_mio, Err, wext = REMEZ_FIR_MLLS(order=N, edge=Be, fx=D, 
                          wtx = W, filter_type = Ftype, 
                          lgrid = lgrid, maxiter=maxit, bDebug= True)

# Calculate resulting passband and stopband ripples
deltac = Err / W[0]
deltas = Err / W[1]
ripple_obtenido = (1 + deltac) / (1 - deltac)

# Normalize the maximum gain to 1
hh_mio = hh_mio / (1 + deltac)

# Design the filter using the Remez algorithm
hh_mio, Err, wext = REMEZ_FIR_MLLS(order=N, edge=Be, fx=D, 
                          wtx = W, filter_type = Ftype, 
                          lgrid = lgrid, maxiter=maxit)

# Normalize the maximum gain to 1
hh_mio = hh_mio / (1 + deltac)

# Output the errors and attenuation values

print(f"Err = {Err:.8f}")
Amax = 20 * np.log10(ripple_obtenido)
Amin = 20 * np.log10((1 - deltac) / deltas)
print(f"Amax = {Amax:.6f} dB")
print(f"Amin = {Amin:.4f} dB")

fs = 2.0
h_firpm = remez(N, Be, [1, 0], weight= W, fs=2.0, maxiter=maxit)

fft_sz = 512
half_fft_sz = fft_sz//2

H_mio = np.fft.fft(hh_mio, fft_sz)
H_tapio = np.fft.fft(hh_tapio, fft_sz)
H_scipy = np.fft.fft(h_firpm, fft_sz)
frecuencias = np.arange(start=0, stop=fs, step=fs/fft_sz )

wextt = (wext * (half_fft_sz-1)).astype(int)

plt.figure()
plt.clf()

# Graficar la respuesta en frecuencia
plt.plot(frecuencias[:half_fft_sz], 20*np.log10(np.abs(H_mio[:half_fft_sz])), label='mi firPM')
plt.plot(frecuencias[wextt], 20*np.log10(np.abs(H_mio[wextt])), 'or', label='$\omega_{ext}$')
# plt.plot(frecuencias[:half_fft_sz], 20*np.log10(np.abs(H_tapio[:len(H_tapio)//2])), label='Tapio')
plt.plot(frecuencias[:half_fft_sz], 20*np.log10(np.abs(H_scipy[:len(H_scipy)//2])), label='Scipy')

plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)

plt.title("Respuesta en Frecuencia del Filtro FIR Diseñado")
plt.xlabel("Frecuencia Normalizada")
plt.ylabel("Magnitud")
plt.legend()
plt.show()



