#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano

Descripción: Script para ejemplificar el uso de filtros digitales FIR e IIR 
estudiados en Teoría de Circuitos II. Se trabaja sobre una señal electrocardiográfica
registrada a 1 kHz, con diversos tipos de contaminación, que se buscan eliminar 
con los filtros diseñados. La plantilla de diseño se corresponde con un filtro
pasabanda con banda de paso de 1 a 35 Hz. Los detalles de la plantilla se pueden
ver en la implementación.

Algunos aspectos a considerar:
-----------------------------
    
    + El diseño de filtros FIR e IIR puede resolverse muy sencillamente con las rutinas incluidas
    en SciPy Signal. 
    
    + Sin embargo, algunas funciones son poco prácticas al momento de escribir estas rutinas.
    Las de filtros IIR funcionan correctamente, mientras que las de FIR dan bastante trabajo
    para cumplir con una especificación estricta, como la que mostramos en este ejemplo.
    
    + Los filtros diseñados, son efectivos para suprimir las interferencias, pero
    ineficaces para ser utilizados en la práctica, debido a la distorsión que
    introducen, especialmente a los tramos de señal no contaminados. Es decir, 
    son efectivos pero no inocuos.
    
    + Los filtros IIR que cumplen plantillas muy estrictas en cuanto a transición y atenuación,
    descuidan las respuestas temporales. Esto conlleva distorsiones de fase muy notorias. 
    Comparar luego de implementar el filtrado bidireccional para neutralizar los efectos
    de la fase no-lineal.

"""

import scipy.signal as sig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import  plot_plantilla, group_delay
from pytc2.filtros_digitales import fir_design_pm, fir_design_ls

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten()
cant_muestras = len(ecg_one_lead)

fs = 1000 # Hz
nyq_frec = fs / 2


# filter design
ripple = 0.5 # dB
atenuacion = 40 # dB

ws1 = 0.1 #Hz
wp1 = 1.0 #Hz
wp2 = 40.0 #Hz
ws2 = 50.0 #Hz



def plot_fir_response(b_coeffs, fir_lbl = 'un_FIR'):

    w_rad  = np.append(np.logspace(-3, 0.8, 1000), np.logspace(0.9, 1.8, 1000) )
    w_rad  = np.append(w_rad, np.linspace(64, nyq_frec, 1000, endpoint=True) ) / nyq_frec * np.pi
    
    _, h_fir = sig.freqz(b_coeffs, worN=w_rad)
    
    w = w_rad / np.pi * nyq_frec
    
    this_lbl = fir_lbl + ' {:d}'.format(b_coeffs.shape[0])
    
    plt.figure(1)
    
    plt.plot(w, 20*np.log10(np.abs(h_fir)+1e-12), label= this_lbl )

    plt.figure(2)
    
    phase_fir = np.angle(h_fir)
    
    plt.plot(w, phase_fir, label= this_lbl)    # Bode phase plot
    
    plt.figure(3)
    
    gd_fir = group_delay(w_rad, phase_fir)
    
    # Para órdenes grandes 
    plt.plot(w[gd_fir > 0], gd_fir[gd_fir>0], label=this_lbl )    # Bode phase plot
    
    

def plot_iir_response(sos_iir, iir_lbl = 'un_IIR'):

    w_rad  = np.append(np.logspace(-3, 0.8, 1000), np.logspace(0.9, 1.8, 1000) )
    w_rad  = np.append(w_rad, np.linspace(64, nyq_frec, 1000, endpoint=True) ) / nyq_frec * np.pi
    
    _, h_iir = sig.sosfreqz(sos_iir, worN=w_rad)
    
    w = w_rad / np.pi * nyq_frec
    
    this_lbl = iir_lbl + ' {:d}'.format(sos_iir.shape[0]*2)
    
    plt.figure(1)
    
    plt.plot(w, 20*np.log10(np.abs(h_iir)+1e-12), label= this_lbl )

        
                
    plt.figure(2)
    
    phase_fir = np.angle(h_iir)
    
    plt.plot(w, phase_fir, label= this_lbl)    # Bode phase plot
    
    plt.figure(3)
    
    gd_fir = group_delay(w_rad, phase_fir)
    
    # Para órdenes grandes 
    plt.plot(w[gd_fir > 0], gd_fir[gd_fir>0], label=this_lbl )    # Bode phase plot
    
    return int(np.round(np.median(gd_fir[np.bitwise_and(w > 3, w < 20)])))



def impulse_response(D, U):
    # Longitud mínima del filtro (por índice máximo 2UD)
    L = 2 * U * D + 1
    h = np.zeros(L)

    h[0] = -1 / D**2
    h[U * (D - 1)] = 1
    h[U * D] = -2 + 2 / D**2
    h[U * (D + 1)] = 1
    h[2 * U * D] = -1 / D**2

    return h

def trim_zeros_edges(x):
    nonzero_indices = np.nonzero(x)[0]
    if nonzero_indices.size == 0:
        return np.array([])  # Todo es cero
    start = nonzero_indices[0]
    end = nonzero_indices[-1]
    return x[start:end + 1]


#%% Config


frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)


cant_coef_hp = 1401   
cant_coef_lp = 171
cant_coef_bp = 1651

demora = (cant_coef_bp - 1)//2
den = 1.0

bPlantillaPlot = False

bIIRbutter = False
bIIRcauer = False
bIIRcheby = False
bScipyLS = False
bScipyRemez = False
bScipyWin = False
bPyTC2remez = False
bPyTC2ls = False


bIIRbutter = True
# bIIRcauer = True
# bIIRcheby = True
# bScipyLS = True
# bScipyRemez = True
# bScipyWin = True
bPyTC2remez = True
bPyTC2ls = True

# Forma de diseñar el pasabanda
# bMethod1 = True # LP + HP
bMethod1 = False # BP

# Filtrado bidireccional
bBidirFilt = True
# bBidirFilt = False
    
#%% Diseño de filtros

#%% IIR SciPy

if bIIRbutter:

    bp_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='butter', output='sos')
    demora_butter = plot_iir_response(bp_sos_butter, iir_lbl = 'Butter')

if bIIRcheby:
    
    bp_sos_cheby = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='cheby1', output='sos')
    demora_cheby = plot_iir_response(bp_sos_cheby, iir_lbl = 'Cheby')


if bIIRcauer:
    
    bp_sos_cauer = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='ellip', output='sos')
    demora_cauer = plot_iir_response(bp_sos_cauer, iir_lbl = 'Cauer')

# SciPy Least-squares

if bScipyLS:

    if bMethod1:
        
        desired = [0, 0, 1, 1]
        weights = [20, 1]
        band_edges = np.array([0.0, ws1, wp1, nyq_frec ])
            
        num_firls_hp = sig.firls(cant_coef_hp, band_edges, desired, weight = np.array([20, 1]), fs = fs)
        
        band_edges = np.array([0.0, wp2, ws2,  nyq_frec ])
        desired = [1, 1, 0, 0]
        weights = [1, 10]
        
        num_firls_lp = sig.firls(cant_coef_lp, band_edges, desired, weight = weights, fs=fs)
        
        num_firls = sig.convolve(num_firls_hp, num_firls_lp)
    
        # num_firls_hp = sig.firls(cant_coef, band_edges, gains[:4], weight = np.array([20, 1]), fs=2)
        
        fir_sz = len(num_firls)
        
        # plot_fir_response(num_firls_hp, fir_lbl=f'SciPy-LS{len(num_firls_hp)}')
        # plot_fir_response(num_firls_lp, fir_lbl=f'SciPy-LS{len(num_firls_hp)}')
        plot_fir_response(num_firls, fir_lbl='SciPy-LS:')

    else:
        
        min_trans = np.min( np.abs([ws1-wp1, wp2-ws2]) ) 
        band_edges = np.array([0.0, wp1-min_trans, wp1, wp2, wp2+min_trans, nyq_frec ])
        desired = [0, 0, 1, 1, 0, 0]
        weights = [1, 1, 1]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        num_firls = sig.firls(cant_coef_bp, band_edges, desired, weight = weights, fs=fs)
    
        fir_sz = len(num_firls)
        
        plot_fir_response(num_firls, fir_lbl=f'PyTC2-Remez{fir_sz}')


# SciPy Remez

if bScipyRemez:

    if bMethod1:

        band_edges = np.array([0.0, ws1, wp1, nyq_frec ])
        desired = [0, 1]
        weights = [10, 1]
        lgrid = 16
           
        num_remez_hp = sig.remez(cant_coef_hp, band_edges, desired, fs=fs, weight = weights, grid_density = lgrid)
        
        band_edges = np.array([0.0, wp2, ws2, nyq_frec ])
        desired = [1, 0]
        weights = [1, 3]
        lgrid = 16
       
        num_remez_lp = sig.remez(cant_coef_lp, band_edges, desired, fs=fs, weight = weights, grid_density = lgrid)
    
        num_remez = sig.convolve(num_remez_hp, num_remez_lp)
    
        fir_sz = len(num_remez)
        
        plot_fir_response(num_remez, fir_lbl=f'SciPy-Remez{fir_sz}')

    else:
    
        min_trans = np.min( np.abs([ws1-wp1, wp2-ws2]) ) 
        band_edges = np.array([0.0, wp1-min_trans, wp1, wp2, wp2+min_trans, nyq_frec ])
        desired = [0, 1, 0]
        weights = [5, 1, 5]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        num_remez = sig.remez(cant_coef_bp, band_edges, desired, fs=fs, weight = weights, grid_density = lgrid)
    
        fir_sz = len(num_remez)
        
        plot_fir_response(num_remez, fir_lbl=f'SciPy-Remez{fir_sz}')


# SciPy Windows

if bScipyWin:

    
    # win_name = 'blackmanharris'
    win_name = 'hann'
    
    if bMethod1:
        
        desired = [0, 0, 1, 1]
        band_edges = np.array([0.0, ws1, wp1, nyq_frec ])
            
        num_win_hp = sig.firwin2(cant_coef_hp, band_edges, desired, fs = fs, window= win_name)
        
        band_edges = np.array([0.0, wp2, ws2,  nyq_frec ])
        desired = [1, 1, 0, 0]
    
        num_win_lp = sig.firwin2(cant_coef_lp, band_edges, desired, fs = fs, window= win_name)
    
        num_win = sig.convolve(num_win_lp, num_win_hp)
    
        fir_sz = len(num_win)
        
        plot_fir_response(num_win, fir_lbl=f'SciPy-Win{fir_sz}')
        # plot_fir_response(num_win_lp, fir_lbl=f'SciPy-Win{fir_sz}')
        # plot_fir_response(num_win_hp, fir_lbl=f'SciPy-Win{fir_sz}')

    else: 
    
        min_trans = np.min( np.abs([ws1-wp1, wp2-ws2]) ) 
        band_edges = np.array([0.0, wp1-min_trans, wp1, wp2, wp2+min_trans, nyq_frec ])
        desired = [0, 0, 1, 1, 0, 0]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        num_win = sig.firwin2(cant_coef_bp, band_edges, desired, fs = fs, window= win_name)
    
        fir_sz = len(num_win)
        
        plot_fir_response(num_win, fir_lbl=f'SciPy-Win{fir_sz}')

# PyTC2 Least-squares

if bPyTC2ls:

    if bMethod1:
        
        band_edges = np.array([0.0, ws1, wp1, nyq_frec ])
        desired = [0, 0, 1, 1]
        weights = [1, 1]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        b_ls_hp = fir_design_ls(cant_coef_hp-1, band_edges, desired, fs = fs, weight = weights, grid_density=lgrid, filter_type='highpass')
        
        band_edges = np.array([0.0, wp2, ws2, nyq_frec ])
        desired = [1, 1, 0, 0]
        weights = [1, 50]
        lgrid = 16
        
        # ojo que paso el orden cant_coef_hp-1
        b_ls_lp = fir_design_ls(cant_coef_lp, band_edges, desired, fs = fs, weight = weights, grid_density=lgrid, filter_type='lowpass')
        
        num_firls_pytc2 = sig.convolve(b_ls_hp, b_ls_lp)
    
        fir_sz = len(num_firls_pytc2)
        
        plot_fir_response(num_firls_pytc2, fir_lbl='PyTC2-LS:')
        # plot_fir_response(b_ls_hp, fir_lbl='PyTC2-LS:')
        # plot_fir_response(b_ls_lp, fir_lbl='PyTC2-LS:')
    
    else: 
        
        min_trans = np.min( np.abs([ws1-wp1, wp2-ws2]) ) 
        band_edges = np.array([0.0, wp1-min_trans, wp1, wp2, wp2+min_trans, nyq_frec ])
        desired = [0, 0, 1, 1, 0, 0]
        weights = [1, 1, 1]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        num_firls_pytc2 = fir_design_ls(cant_coef_bp, band_edges, desired, fs = fs, weight= weights, grid_density=lgrid, filter_type='bandpass')
    
        fir_sz = len(num_firls_pytc2)
        
        plot_fir_response(num_firls_pytc2, fir_lbl=f'PyTC2-Remez{fir_sz}')


    
# PyTC2 Remez


if bPyTC2remez:

    if bMethod1:

        # Un enfoque: partir el diseño BP asimétrico en HP en cascada con LP.
        band_edges = np.array([0.0, ws1, wp1, nyq_frec ])
        desired = [0, 0, 1, 1]
        weights = [2, 1]
        lgrid = 16
            
        # ojo que paso el orden cant_coef_hp-1
        b_pm_hp, _, _ = fir_design_pm(cant_coef_hp-1, band_edges, desired, fs = fs, weight= weights, grid_density=lgrid, filter_type='highpass')
        
        band_edges = np.array([0.0, wp2, ws2, nyq_frec ]) 
        desired = [1, 1, 0, 0]
        weights = [2, 5]
        lgrid = 16
    
        # ojo que paso el orden cant_coef_hp-1
        b_pm_lp, _, _ = fir_design_pm(cant_coef_lp-1, band_edges, desired, weight= weights, fs = fs, grid_density=lgrid, filter_type='lowpass')
    
        num_remez_pytc2 = sig.convolve(b_pm_hp, b_pm_lp)

        plot_fir_response(num_remez_pytc2, fir_lbl=f'PyTC2-Remez{fir_sz}')
        # plot_fir_response(b_pm_hp, fir_lbl=f'PyTC2-Remez{fir_sz}')
        # plot_fir_response(b_pm_lp, fir_lbl=f'PyTC2-Remez{fir_sz}')

    else:
        
        # Otro enfoque, hacer simétrica la plantilla
        min_trans = np.min( np.abs([ws1-wp1, wp2-ws2]) ) 
        band_edges = np.array([0.0, wp1-min_trans, wp1, wp2, wp2+min_trans, nyq_frec ])
        desired = [0, 0, 1, 1, 0, 0]
        weights = [5, 1, 5]
        lgrid = 16
        cant_coef_bp = 1650
            
        # ojo que paso el orden cant_coef_hp-1
        num_remez_pytc2, _, _ = fir_design_pm(cant_coef_bp, band_edges, desired, fs = fs, weight= weights, grid_density=lgrid, filter_type='bandpass')
    
    
        fir_sz = len(num_remez_pytc2)
        
        plot_fir_response(num_remez_pytc2, fir_lbl=f'PyTC2-Remez{fir_sz}')
    

# Plantilla y finalización de figuras.

plt.figure(1)

plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.axis([0, 500, -60, 5 ]);

axes_hdl = plt.gca()
axes_hdl.legend()

plt.grid()

            
plt.figure(2)

plt.title('FIR diseñado Fase')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Fase [rad]')

axes_hdl = plt.gca()
axes_hdl.legend()


plt.figure(3)

plt.axis([0, 500, 0, cant_coef_bp*3/2])

plt.title('FIR diseñado retardo')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Retardo [s]')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()


#%% Ahora filtramos con cada filtro diseñado


if bIIRbutter:
    if bBidirFilt:
        ECG_f_butt = sig.sosfiltfilt(bp_sos_butter, ecg_one_lead)
    else:
        ECG_f_butt = sig.sosfilt(bp_sos_butter, ecg_one_lead)


if bIIRcauer:

    if bBidirFilt:
        ECG_f_cauer = sig.sosfiltfilt(bp_sos_cauer, ecg_one_lead)
    else:
        ECG_f_cauer = sig.sosfilt(bp_sos_cauer, ecg_one_lead)
    
if bIIRcheby:
    
    if bBidirFilt:
        ECG_f_cheb = sig.sosfiltfilt(bp_sos_cheby, ecg_one_lead)
    else:
        ECG_f_cheb = sig.sosfilt(bp_sos_cheby, ecg_one_lead)
        

if bScipyLS:
    
    if bBidirFilt:
        ECG_f_ls = sig.filtfilt(num_firls, den, ecg_one_lead)
    else:
        ECG_f_ls = sig.lfilter(num_firls, den, ecg_one_lead)


if bScipyRemez:
    
    if bBidirFilt:
        ECG_f_remez = sig.filtfilt(num_remez, den, ecg_one_lead)
    else:
        ECG_f_remez = sig.lfilter(num_remez, den, ecg_one_lead)

if bScipyWin:
    
    if bBidirFilt:
        ECG_f_win = sig.filtfilt(num_win, den, ecg_one_lead)
    else:
        ECG_f_win = sig.lfilter(num_win, den, ecg_one_lead)

if bPyTC2remez:

    if bBidirFilt:
        ECG_f_pytc2_remez = sig.filtfilt(num_remez_pytc2, den, ecg_one_lead)
    else:
        ECG_f_pytc2_remez = sig.lfilter(num_remez_pytc2, den, ecg_one_lead)
    
if bPyTC2ls:

    if bBidirFilt:
        ECG_f_pytc2_LS = sig.filtfilt(num_firls_pytc2, den, ecg_one_lead)
    else:
        ECG_f_pytc2_LS = sig.lfilter(num_firls_pytc2, den, ecg_one_lead)



regs_interes = ( 
        np.array([2, 2.2]) *60*fs, # minutos a muestras
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([10, 10.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        np.array([18, 18.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)


    if bIIRbutter:
        
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
        else:
            plt.plot(zoom_region, ECG_f_butt[zoom_region+ demora_butter], label='Butter')

    
    if bIIRcauer:
    
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer')
        else:
            plt.plot(zoom_region, ECG_f_cauer[zoom_region+ demora_cauer], label='Cauer')


    if bIIRcheby:
        
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_cheb[zoom_region], label='Cheby')
        else:
            plt.plot(zoom_region, ECG_f_cheb[zoom_region+ demora_cheby], label='Cheby')

    
    if bScipyLS:
        
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_ls[zoom_region], label='LS')
        else:
            plt.plot(zoom_region, ECG_f_ls[zoom_region+demora], label='LS')
        
    
    if bScipyRemez:
        
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_remez[zoom_region], label='Remez')
        else:
            plt.plot(zoom_region, ECG_f_remez[zoom_region+demora], label='Remez')

    if bScipyWin:
        
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_win[zoom_region], label='Win')
        else:
            plt.plot(zoom_region, ECG_f_win[zoom_region+demora], label='Win')
       
    if bPyTC2remez:
    
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_pytc2_remez[zoom_region], label='PyTC2-Remez')
        else:
            plt.plot(zoom_region, ECG_f_pytc2_remez[zoom_region+demora], label='PyTC2-Remez')

        
    if bPyTC2ls:
    
        if bBidirFilt:
            plt.plot(zoom_region, ECG_f_pytc2_LS[zoom_region], label='PyTC2-LS')
        else:
            plt.plot(zoom_region, ECG_f_pytc2_LS[zoom_region+demora], label='PyTC2-LS')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
            
    plt.show()



