#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 07:21:50 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import  plot_plantilla, group_delay
from pytc2.filtros_digitales import fir_design_pm, fir_design_ls
import scipy.signal as sig

def plot_ifir_response(h_fir, w ,  fir_lbl = 'un_FIR'):
    
    plt.figure(1)
    
    plt.plot(w / np.pi * nyq_frec, 20*np.log10(np.abs(h_fir)+1e-12), label= fir_lbl )

    plt.figure(2)
    
    phase_fir = np.unwrap(np.angle(h_fir))
    
    plt.plot(w/ np.pi * nyq_frec, phase_fir, label= fir_lbl)    # Bode phase plot
    
    plt.figure(3)
    
    gd_fir = group_delay(w, phase_fir)
    
    # Para órdenes grandes 
    plt.plot(w[gd_fir > 0]/ np.pi * nyq_frec, gd_fir[gd_fir>0], label=fir_lbl )    # Bode phase plot
    


def l_phase_bp_fir_order(wT_edge, d, ripple_in_db=False):
    """
    Estimates the required order for a linear-phase bandpass FIR filter.
    
    Inputs:
        wT_edge = [ws1T, wc1T, wc2T, ws2T]
            Normalized band edges (0 < wT_edge < pi) in radians
        d = [ds1, dc, ds2] or [A_min1, A_max, A_min2] if ripple_in_db=True
            ds1: stopband ripple (left)
            dc: passband ripple
            ds2: stopband ripple (right)
            OR
            A_min1: minimum stopband attenuation (left, dB)
            A_max: maximum passband ripple (dB)
            A_min2: minimum stopband attenuation (right, dB)
        ripple_in_db: boolean indicating if d is provided in dB (default False)
    
    Outputs:
        N: Filter order estimate
        Be: Band edges [0, ws1T, wc1T, wc2T, ws2T, pi]
        D: Desired values [0, 0, 1, 1, 0, 0]
        W: Weighting factors [dc/ds1, 1, dc/ds2]
    
	%		Formated for direct use by EQR_FIR.m and firmp.m or remez.m
 
	% 	Toolbox for DIGITAL FILTERS USING MATLAB
	
	%	Author: 			Lars Wanhammar 2003-07-29
	%	Modfied by:			LW 2004-09-22, 2014-11-17
	% 	Copyright:			by authors - not released for commercial use
	%	Version: 			1 
	%	Known bugs:			
	% 	Report bugs to:		llamedom@gmail.com    
    
    Reference: Mintzer F. and Liu B.: Practical design rules for optimum FIR 
               bandpass digital filters, IEEE Trans. on Acoustics, Speech and 
               Signal Processing, Vol. ASSP-27, No. 2, pp. 204-206, April 1979.
    """
    if not (wT_edge[0] < wT_edge[1] and wT_edge[1] < wT_edge[2] and wT_edge[2] < wT_edge[3]):
        raise ValueError('Improper band edges. Edges should be: ws1T < wc1T < wc2T < ws2T.')
    
    # Convert from dB to linear if needed
    if ripple_in_db:
        A_min1, A_max, A_min2 = d
        # Calculate delta_c from A_max
        delta_c = (10**(A_max/20) - 1) / (10**(A_max/20) + 1)
        # Calculate delta_s from A_min
        delta_s1 = (1 + delta_c) / 10**(A_min1/20)
        delta_s2 = (1 + delta_c) / 10**(A_min2/20)
        ds1, dc, ds2 = delta_s1, delta_c, delta_s2
    else:
        ds1, dc, ds2 = d
    
    DwT = min(wT_edge[1] - wT_edge[0], wT_edge[3] - wT_edge[2]) / (2 * np.pi)
    DwTmax = max(wT_edge[1] - wT_edge[0], wT_edge[3] - wT_edge[2]) / (2 * np.pi)
    R = DwTmax / DwT
    
    xc = np.log10(dc)
    xs = np.log10(min(ds1, ds2))
    
    b1 = 0.01201 
    b2 = 0.09664 
    b3 = -0.51325 
    b4 = 0.00203
    b5 = -0.57054 
    b6 = -0.44314
    
    C = ((b1 * xc + b2) * xc + b3) * xs + (b4 * xc + b5) * xc + b6
    G = 14.6 * (xc - xs) + 16.9
    
    N = np.floor((C / DwT + G * DwT + 1) * (1 - 0.034 * (R - 2)))
    
    Be = np.array([0, wT_edge[0], wT_edge[1], wT_edge[2], wT_edge[3], np.pi])
    D = np.array([0, 0, 1, 1, 0, 0])
    W = np.array([dc/ds1, 1, dc/ds2])
    
    return N, Be, D, W

def herrmann_lp_fir_order(wT, d, ripple_in_db=False):
    """
    Estimación de Herrmann del orden N para filtros FIR pasa bajos de fase lineal (minimax).
    
    Parámetros:
        wT = [wcT, wsT]
            wcT: passband cutoff edge (NORMALIZED 0 < wcT < pi)
            wsT: stopband cutoff edge > wcT 
        d = [dc, ds] or [A_max, A_min] depending on ripple_in_db
            dc: passband ripple (linear)
            ds: stopband ripple (linear)
            OR
            A_max: maximum passband ripple (dB)
            A_min: minimum stopband attenuation (dB)
        ripple_in_db: boolean indicating if d is provided in dB (default False)
    
    Outputs:
        N: Filter order estimate
        Be: Band edges for use with remez function [0, wcT, wsT, pi]
        D: Desired values at band edges [1, 1, 0, 0]
        W: Weighting factors [1, dc/ds]
        
		Formated for direct use by MPR_FIR.m and firmp.m or remez.m	
	
	 	Toolbox for DIGITAL FILTERS USING MATLAB
	
	 	Author: 		Lars Wanhammar 2004-07-17
	 	Modified by: 	LW 2005-05-09
	 	Copyright:		by authors - not released for commercial use
	 	Version:		1 
	 	Known bugs:	 
	 	Report bugs to:	Wanhammar@gmail.com
	
	 	References:	[1] Herrmann O., Rabiner L.R., and Chan D S K.:  Practical 
	 					design rules for optimum finite impulse response lowpass digital 
	 					filters, Bell System Techical Journal, vol. 52 (July-August), 1973.
					[2] Rabiner & Gold, Theory and Appications of DSP, pp. 156-7.             
        
    """
    
    if wT[0] >= wT[1] or wT[0] >= np.pi or wT[1] > np.pi:
        raise ValueError('Improper band edges. Edges should be: wcT < wsT <= pi.')
    
    # Convert from dB to linear if needed
    if ripple_in_db:
        A_max, A_min = d
        # Calculate delta_c from A_max
        delta_c = (10**(A_max/20) - 1) / (10**(A_max/20) + 1)
        # Calculate delta_s from A_min and delta_c
        delta_s = (1 + delta_c) / 10**(A_min/20)
        d = [delta_c, delta_s]
    
    
    wcT, wsT = wT
    if wcT >= wsT or wcT >= np.pi or wsT > np.pi:
        raise ValueError("Improper band edges. Edges should be: wcT < wsT <= π.")

    dwT = wsT - wcT
    dc, ds = d

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

    # If the passband ripple is smaller, then interchange the roles of the
    # ripples; comment by Tapio - for sure, this can be found [1] after careful
    # reading
    # Si el ripple del pasabanda es menor que el del stopband, invertirlos
    if dc < ds:
        d2 = np.log10(dc)
        d1 = np.log10(ds)
    else:
        d1 = ldc
        d2 = lds

    F = b1 + b2 * (d1 - d2)
    D = (a1 * d1**2 + a2 * d1 + a3) * d2 + (a4 * d1**2 + a5 * d1 + a6)
    Tpi = 2 * np.pi
    N = int(np.ceil(Tpi * D / dwT - F * dwT / Tpi))

    Be = np.array([0, wcT, wsT, np.pi])
    D_vec = np.array([1, 1, 0, 0])
    W = np.array([1, dc / ds])

    return N, Be, D_vec, W

def upsample(bFIR, L):
    """
    Intercala L-1 ceros entre cada muestra de la respuesta al impulso b_G.
    
    Parámetros:
        b_G: array con la respuesta al impulso del filtro
        L: factor de interpolación (se insertarán L-1 ceros entre muestras)
    
    Retorna:
        Array con los ceros intercalados
    """
    # Crear un array de ceros con el tamaño adecuado
    resultado = np.zeros(len(bFIR) * L)
    
    # Insertar las muestras originales en las posiciones cada L muestras
    resultado[::L] = bFIR
    
    return resultado

plt.close('all')

# Ejemplo para ECG @ fs
fs = 1000 # Hz
nyq_frec = fs/2

ws1 = 0.1 #Hz
wp1 = 1.0 #Hz
wp2 = 35.0 #Hz
ws2 = 36.0 #Hz


wbands = np.array([ ws1, wp1, wp2, ws2 ]) * 2*np.pi/fs

M_max = int((nyq_frec) // ws2 )
# atenuación
Amax = .5 # dB
Amin = 40 # dB

Ng = np.zeros(M_max-1)  # índice 0 no se usa
Nf = np.zeros(M_max-1)

all_M = np.arange(2, M_max+1)

ii = 0
# Bucle sobre M
for M in all_M:
    Ng[ii], _, _, _ = l_phase_bp_fir_order(wbands * M, [Amin, Amax, Amin], ripple_in_db=True)
    Nf[ii], _, _, _ = herrmann_lp_fir_order([wbands[2], 2 * np.pi / M - wbands[3]], [Amax, Amin], ripple_in_db=True)
    ii += 1

Ntot = Ng + Nf
nidx = np.argmin(Ntot)

M_opt = all_M[nidx]

# Gráfico
plt.figure(4)
plt.clf()
plt.plot(all_M , Ng, '--', linewidth=2, label='Ng')
plt.plot(all_M , Nf, ':', linewidth=2, label='Nf')
plt.plot(all_M , Ntot, '-', linewidth=2, label='Ng + Nf')
plt.plot(all_M[nidx], Ntot[nidx], 'o', linewidth=2, label='$M_{OPT}$')

plt.grid(True)
plt.xlabel(r'$M$', fontsize=16)
plt.ylabel('Number of multiplications', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.axis([2, 19, 0, 600])

plt.legend()
plt.tight_layout()
plt.show()


# 

# Un enfoque: partir el diseño BP asimétrico en HP en cascada con LP.
wbands = np.array([ ws1, wp1, wp2, ws2 ]) * 2*np.pi/fs
lgrid = 16

cant_coef, band_edges, desired, weights = l_phase_bp_fir_order(wbands * M_opt, [Amin, Amax, Amin], ripple_in_db=True)
    
# ojo que paso el orden cant_coef_hp-1
b_G, _, _ = fir_design_pm(cant_coef, band_edges, desired, fs = 2*np.pi, weight= weights, grid_density=lgrid, filter_type='bandpass', max_iter = 100)

cant_coef, band_edges, desired, weights = herrmann_lp_fir_order([wbands[2], 2 * np.pi / M_opt - wbands[3]], [Amax, Amin], ripple_in_db=True)

# ojo que paso el orden cant_coef_hp-1
b_F, _, _ = fir_design_pm(cant_coef, band_edges, desired, fs = 2*np.pi, weight= weights, grid_density=lgrid, filter_type='lowpass')


# b_Gu = upsample(b_G, M_opt)
# b_ifir = sig.convolve(b_Gu, b_F)


w_rad  = np.append(np.logspace(-3, 0.8, 1000), np.logspace(0.9, 1.8, 1000) )
w_rad  = np.append(w_rad, np.linspace(64, nyq_frec, 1000, endpoint=True) ) / nyq_frec * np.pi

_, h_G = sig.freqz(b_G, worN=w_rad*M_opt)


w_rad, h_F = sig.freqz(b_F, worN=w_rad)

H_iFIR = h_G * h_F

# w_rad  = w_rad / np.pi * nyq_frec

fir_sz = len(b_G) + len(b_F)
fir_sz = M_opt*len(b_G) + len(b_F)

# plot_fir_response(b_G, fir_lbl=f'bG')
# plot_fir_response(b_F, fir_lbl=f'bF')
# plot_fir_response(b_Gu, fir_lbl=f'bGu')
plot_ifir_response(H_iFIR, w_rad, fir_lbl=f'PyTC2-IFIR:{fir_sz}')


plt.figure(1)

frecs = np.array([ ws1, wp1, wp2, ws2 ])

plot_plantilla(filter_type = 'bandpass', fpass = frecs[[1, 2]], ripple = Amax , fstop = frecs[[0, 3]], attenuation = Amin, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
# plt.axis([0, 500, -50, 5 ]);

axes_hdl = plt.gca()
axes_hdl.legend()

plt.grid()

