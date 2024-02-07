#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:58:29 2024

@author: mariano
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

#%% Implementación en Python simil RT

def promediador_rt_init( xx, DD, UU ):

    # ventana de selección de UU muestras por el sobremuestreo
    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)
    
    # se asume como salida el mismo valor medio para las primeras UU muestras
    yy_ci = np.zeros(UU)
    yy_ci[:] = np.sum( xx[:(DD * UU)] * hh_u)

    # se consideran las primeras DD muestras a una frec de muestreo UU veces más
    # elevada.
    xx_ci = xx[:(DD * UU) ]

    return( (xx_ci, yy_ci) )

def promediador_rt( xx, DD, UU, xx_ci, yy_ci, kk_offset = 0):
    
    NN = xx.shape[0]

    # resultaron ser importante las condiciones iniciales
    yy = np.zeros_like(xx)
    # yy = np.ones_like(xx) * xx[0] * DD * UU

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
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)

        # inicio de la recursión
        for kk in range(DD * UU, (DD * UU) + UU ):
    
            ii = kk-1
            # Calcula la salida según la ecuación recursiva
            yy[ii] = np.sum(xx[kk-(DD * UU):kk] * hh_u)

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

def filtro_peine_DCyArmonicas( xx, DD = 16, UU = 2, MA_stages = 2 ):

    NN = xx.shape[0]

    ###############################################################################
    # estimación del valor medio en ventanas de DD muestras y sobremuestreo por UU

    # Se plantea la posibilidad de una implementación en un entorno de memoria
    # limitada que obligue al procesamiento de "bloque a bloque"
    
    # se calculan condiciones iniciales para el primer bloque moving averager (MA)
    # en total habrá MA_stages en cascada.
    xx_ci, yy_ci = promediador_rt_init( xx, DD, UU )

    yy = np.zeros_like(xx)
    
    # se procesa cada bloque por separado y se concatena la salida
    for jj in range(0, NN, block_s):
    
        yy_aux, xx_ci, yy_ci = promediador_rt( xx[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)

        yy[jj:jj+block_s] = yy_aux

    # cascadeamos MA_stages-1 más
    for ii in range(1, MA_stages):

        # se calculan condiciones iniciales
        xx_ci, yy_ci = promediador_rt_init( yy, DD, UU )
        
        for jj in range(0, NN, block_s):
        
            yy_aux, xx_ci, yy_ci = promediador_rt( yy[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
        
            yy[jj:jj+block_s] = yy_aux

    #############################################################
    # demora de la señal xx y resta de la salida del último MA
    
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU) )
    yy = xx_aux - yy
    return( yy )
       
#%% Pruebas con ECG real

fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2

mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten().astype(np.float64)
cant_muestras = len(ecg_one_lead)

# bloque de memoria para la implementación RT
block_s = cant_muestras//4
# block_s = cant_muestras

dd = 64
uu = 20
ma_st = 2

# demora teórica del filtro de Rick
demora_rl = int((dd-1)/2*ma_st*uu)

ECG_f_rl_fin = filtro_peine_DCyArmonicas( ecg_one_lead, DD = dd, UU = uu, MA_stages = ma_st )

# Nota: se obtuvo esta performance en una PC de escritorio estandard con:
# Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
# RAM: 8GB
# Manufacturer: Gigabyte Technology Co., Ltd.
# Product Name: B85M-D3H
# 1129116 muestras de ECG a fs = 1kHz
# %timeit ECG_f_rl_fin = filtro_peine_DCyArmonicas( ecg_one_lead, DD = dd, UU = uu, MA_stages = ma_st )
# 2.01 s ± 73.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

plt.close('all')

regs_interes = ( 
        
        # np.array([-2000, 2000]) + block_s, # minutos a muestras
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

    # FIR con corrección de demora
    plt.plot(zoom_region, ECG_f_rl_fin[zoom_region+demora_rl], ':x', alpha=0.5, label='final')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
            
    plt.show()


    
