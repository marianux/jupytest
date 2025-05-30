#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mariano

Descripción: Script para ejemplificar el uso de filtros digitales FIR e IIR 
estudiados en Teoría de Circuitos II. Se trabaja sobre una señal electrocardiográfica
registrada a 1 kHz, con diversos tipos de contaminación, que se buscan eliminar 
con los filtros diseñados. La plantilla de diseño se corresponde con un filtro
pasabanda con banda de paso de 3 a 25 Hz. Los detalles de la plantilla se pueden
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



def plot_fir_response(b_coeffs, w_rad, fir_lbl = 'un_FIR'):

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
    



#%%


frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)


# bp_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='butter', output='sos')
# bp_sos_cheby = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='cheby1', output='sos')
# bp_sos_cauer = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='ellip', output='sos')

cant_coef = 2000

if cant_coef % 2 == 0:
    cant_coef += 1
    

band_edges = np.array([0.0, ws1, wp1, nyq_frec ]) / nyq_frec
    
# num_firls_hp = sig.firls(cant_coef, band_edges, gains[:4], weight = np.array([20, 1]), fs=2)
num_remez_hp = sig.remez(cant_coef, band_edges, [gains[1], 1.], weight = np.array([20, 1]), grid_density = 64, fs=2)

desired = [0, 0, 1, 1]
weights = [5, 1]
lgrid = 32

# Diseño del filtro con PM
b_pm_hp, _, _ = fir_design_pm(501, band_edges, desired, weight= weights, grid_density=lgrid, filter_type='highpass', debug=True)

# Diseño del filtro con LS
# b_ls_hp = fir_design_ls(cant_coef, band_edges, gains[:4], grid_density=lgrid, filter_type='highpass')



cant_coef = 500

if cant_coef % 2 == 0:
    cant_coef += 1

# num_firls_lp = sig.firls(cant_coef, np.append( [0.0], frecs[3:]), gains[2:], weight = np.array([5, 10]), fs=2)
num_remez_lp = sig.remez(cant_coef, np.append( [0.0], frecs[3:]), [1., gains[5]], weight = np.array([1, 5]), grid_density = 16, fs=2)

band_edges = np.array([0.0, wp2, ws2, nyq_frec ]) / nyq_frec
desired = [1, 1, 0, 0]
lgrid = 16

# Diseño del filtro con PM
b_pm_lp, _, _ = fir_design_pm(cant_coef, band_edges, desired, grid_density=lgrid, filter_type='lowpass')

# Diseño del filtro con LS
# b_ls_lp = fir_design_ls(cant_coef, band_edges, desired, grid_density=lgrid, filter_type='lowpass')


# num_firls = sig.convolve(num_firls_hp, num_firls_lp)
num_remez = sig.convolve(num_remez_hp, num_remez_lp)

# num_firls_pytc2 = sig.convolve(b_ls_lp, b_ls_hp)
num_remez_pytc2 = sig.convolve(b_pm_lp, b_pm_hp)


# num_win =   sig.firwin2(cant_coef, frecs, gains , window='blackmanharris' )

## Rick Lyons high pass with DC-block filter.
# dd = 16;
# num_rl = np.hstack([-1/dd**2, np.zeros(dd-2), 1, (2/dd**2-2), 1, np.zeros(dd-2), -1/dd**2])
# den_rl = np.array([1, -2, 1])

## Rick Lyons ECG filter
# dd = 16
# uu = 4
# # num_rl = np.hstack([-1/dd, np.zeros(uu*(dd-1)-1), 1, np.zeros(uu-1), (2/dd**2-2), np.zeros(uu-1), 1, np.zeros(uu*(dd-1)-1), -1/dd**2])
# num_rl = np.hstack([-1/dd, np.zeros(uu*(dd-1)-1), 1, np.zeros(uu-1), (2/dd**2-2), np.zeros(uu-1), 1, np.zeros(uu*(dd-1)-1), -1/dd**2])
# den_rl = np.hstack([1, np.zeros(uu-1), -2, np.zeros(uu-1), 1])

# num_rl = np.hstack([-1/dd**2, np.zeros(uu*(dd-1)-1), 1, np.zeros(uu-1), (2/dd**2-2), np.zeros(uu-1), 1, np.zeros(uu*(dd-1)-1), -1/dd**2])
# den_rl = np.hstack([1, np.zeros(uu-1), -2, np.zeros(uu-1), 1])

# demora_rl = int(uu*(dd-1))

demora = int((num_remez.shape[0]-1)/2)
# demora = int((num_remez_pytc2.shape[0]-1)/2)

den = 1.0

w_rad  = np.append(np.logspace(-2, 0.8, 1000), np.logspace(0.9, 1.8, 1000) )
w_rad  = np.append(w_rad, np.linspace(51, nyq_frec, 1000, endpoint=True) ) / nyq_frec * np.pi
# _, h_butter = sig.sosfreqz(bp_sos_butter, w_rad)

# # w_rad, h_butter = sig.sosfreqz(bp_sos_butter, 1024)
# _, h_cheby = sig.sosfreqz(bp_sos_cheby, w_rad)
# _, h_cauer = sig.sosfreqz(bp_sos_cauer, w_rad)
# _, hh_firls = sig.freqz(num_firls, den, w_rad)
# _, hh_remez = sig.freqz(num_remez, den, w_rad)
# _, hh_firls_pytc2 = sig.freqz(num_firls_pytc2, den, w_rad)
# _, hh_remez_pytc2 = sig.freqz(num_remez_pytc2, den, w_rad)


# _, hh_remez_hp = sig.freqz(num_remez_hp, den, w_rad)
# _, hh_remez_lp = sig.freqz(num_remez_lp, den, w_rad)
# _, hh_win = sig.freqz(num_win, den, w_rad)
# _, hh_rl = sig.freqz(num_rl, den_rl, w_rad)

w = w_rad / np.pi * nyq_frec

plt.close('all')

# plot_fir_response(num_firls, w_rad, fir_lbl='FIR-ls')
# plot_fir_response(num_remez_hp, w_rad, fir_lbl='FIR-pm-HP')
# plot_fir_response(num_remez_lp, w_rad, fir_lbl='FIR-pm-LP')
plot_fir_response(num_remez, w_rad, fir_lbl='FIR-remez')
# plot_fir_response(num_firls_pytc2, w_rad, fir_lbl='FIR-ls-pytc2')
plot_fir_response(num_remez_pytc2, w_rad, fir_lbl='FIR-pm-pytc2')
plot_fir_response(b_pm_hp, w_rad, fir_lbl='FIR-pm-HP')
plot_fir_response(b_pm_lp, w_rad, fir_lbl='FIR-pm-LP')
# plot_fir_response(num_win, w_rad, fir_lbl='FIR-Win')

# plt.plot(w, 20*np.log10(np.abs(h_butter)+1e-12), label='IIR-Butter {:d}'.format(bp_sos_butter.shape[0]*2) )
# # plt.plot(w, 20*np.log10(np.abs(h_cheby)+1e-12), label='IIR-Cheby {:d}'.format(bp_sos_cheby.shape[0]*2) )
# # plt.plot(w, 20*np.log10(np.abs(h_cauer)+1e-12), label='IIR-Cauer {:d}'.format(bp_sos_cauer.shape[0]*2) )
# # plt.plot(w, 20 * np.log10(abs(hh_firls)), label='FIR-ls {:d}'.format(num_firls.shape[0]))
# # plt.plot(w, 20 * np.log10(abs(hh_remez)), label='FIR-remez {:d}'.format(num_remez.shape[0]))
# plt.plot(w, 20 * np.log10(abs(hh_firls_pytc2)), label='FIR-ls-pytc2 {:d}'.format(num_firls_pytc2.shape[0]))
# plt.plot(w, 20 * np.log10(abs(hh_remez_pytc2)), label='FIR-remez-pytc2 {:d}'.format(num_remez_pytc2.shape[0]))
# # plt.plot(w, 20 * np.log10(abs(hh_remez_hp)), label='FIR-remez-HP {:d}'.format(num_remez_hp.shape[0]))
# # plt.plot(w, 20 * np.log10(abs(hh_remez_lp)), label='FIR-remez-LP {:d}'.format(num_remez_lp.shape[0]))
# plt.plot(w, 20 * np.log10(abs(hh_win)), label='FIR-Win {:d}'.format(num_win.shape[0]))
# # plt.plot(w, 20 * np.log10(abs(hh_rl)), label='FIR-Rick {:d}'.format(num_rl.shape[0]))

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

plt.axis([0, 500, 0, 1.5*demora ])

plt.title('FIR diseñado retardo')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Retardo [s]')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

#%% Ahora filtramos con cada filtro diseñado


demora_butter = int(np.round(np.median(gd_butter[np.bitwise_and(w > 3, w < 20)])))
demora_cheby = int(np.round(np.median(gd_cheby[np.bitwise_and(w > 3, w < 20)])))
demora_cauer = int(np.round(np.median(gd_cauer[np.bitwise_and(w > 3, w < 20)])))


# FIltrado convencional

ECG_f_butt = sig.sosfilt(bp_sos_butter, ecg_one_lead)
ECG_f_cheb = sig.sosfilt(bp_sos_cheby, ecg_one_lead)
ECG_f_cauer = sig.sosfilt(bp_sos_cauer, ecg_one_lead)

ECG_f_ls = sig.lfilter(num_firls, den, ecg_one_lead)
ECG_f_remez = sig.lfilter(num_remez, den, ecg_one_lead)
ECG_f_win = sig.lfilter(num_win, den, ecg_one_lead)
# ECG_f_rl = sig.lfilter(num_rl, den_rl, ecg_one_lead)


# # FIltrado bidireccional

# ECG_f_butt = sig.sosfiltfilt(bp_sos_butter, ecg_one_lead)
# ECG_f_cheb = sig.sosfiltfilt(bp_sos_cheby, ecg_one_lead)
# ECG_f_cauer = sig.sosfiltfilt(bp_sos_cauer, ecg_one_lead)

# ECG_f_ls = sig.filtfilt(num_firls, den, ecg_one_lead)
# ECG_f_remez = sig.filtfilt(num_remez, den, ecg_one_lead)
# ECG_f_win = sig.filtfilt(num_win, den, ecg_one_lead)
# # ECG_f_rl = sig.filtfilt(num_rl, den_rl, ecg_one_lead)





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

    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECG_f_cheb[zoom_region], label='Cheby')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer')
    # plt.plot(zoom_region, ECG_f_remez[zoom_region], label='Remez')
    # plt.plot(zoom_region, ECG_f_ls[zoom_region], label='LS')
    # plt.plot(zoom_region, ECG_f_win[zoom_region], label='Win')
    
    # FIR con corrección de demora
    # plt.plot(zoom_region, ECG_f_butt[zoom_region+ demora_butter], label='Butter')
    # plt.plot(zoom_region, ECG_f_cheb[zoom_region+ demora_cheby], label='Cheby')
    # plt.plot(zoom_region, ECG_f_cauer[zoom_region+ demora_cauer], label='Cauer')
    # plt.plot(zoom_region, ECG_f_remez[zoom_region+demora], label='Remez')
    # plt.plot(zoom_region, ECG_f_ls[zoom_region+demora], label='LS')
    # plt.plot(zoom_region, ECG_f_win[zoom_region+demora], label='Win')
    # plt.plot(zoom_region, ECG_f_rl[zoom_region+demora_rl], label='Rick')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
            
    plt.show()



