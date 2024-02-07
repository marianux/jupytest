#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este script es uno preliminar a la implementación del filtro peine de RL.
Tiene muchos puntos de depuración para la validación muesrta a muestra del
filtro.

Ver filtro_peine_rick_lyons_ECG.py para la implementación final.

Created on Wed Feb  7 08:57:47 2024

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#%% Implementación en Python simil RT

## Implementación Python
def one_MA_stage_clasica( xx, DD, UU):
    
    NN = xx.shape[0]
    # buffer_size = DD * UU

    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)

    # resultaron ser importante las condiciones iniciales
    yy = np.zeros_like(xx)
    # yy = np.ones_like(xx) * xx[0] * DD * UU

    yy[0:(DD * UU)] = np.sum(xx[0:(DD * UU)] * hh_u)

    # debug muestra a muestra
    if bDebugsbs:
        kk = DD * UU

        print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( kk-1, kk-(DD * UU), kk-1 ))
        xx_aux = xx[kk-(DD * UU):kk] * hh_u
        strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
        strAux = ' + '.join(strAux)
        strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
        print( strAux )
    
    for kk in range(DD * UU,NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk-1] = np.sum(xx[kk-(DD * UU):kk] * hh_u)

        # debug muestra a muestra
        if bDebugsbs:
            if (kk % NN) <= (2*DD * UU):
                print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( kk-1, kk-(DD * UU), kk-1 ))
                xx_aux = xx[kk-(DD * UU):kk] * hh_u
                strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
                strAux = ' + '.join(strAux)
                strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
                print( strAux )
    
    kk += 1
    yy[kk-1] = np.sum(xx[kk-(DD * UU):kk] * hh_u)
    
    # escalo y devuelvo
    return( yy )

## Implementación Python para RT con código de debug
def MA_st_rt_dbg( xx, DD, UU, xx_ci, yy_ci, kk_offset = 0):
    
    NN = xx.shape[0]

    # resultaron ser importante las condiciones iniciales
    yy = np.zeros_like(xx)
    # yy = np.ones_like(xx) * xx[0] * DD * UU

    # primera_xx_ci_idx = (NN - ((DD * UU)+UU))
    primera_xx_ci_idx = (NN - DD * UU)
    primera_yy_ci_idx = (NN - UU)

    if bDebugsbs:
        
        if kk_offset <= (2*DD * UU):
            print('MA rec RT')
            print('---------')

    if(kk_offset == 0):

        # condiciones iniciales
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy_ci[kk]
    
            if bDebugsbs:
                
                if kk <= (2*DD * UU):
                    print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx_ci[{:d}] ({:3.3f}) + yy_ci[{:d}] ({:3.3f})'.format( kk, yy[kk], kk, xx[kk], kk, xx_ci[kk], kk, yy_ci[kk] ))
              
        # extiendo las salidas al mismo valor que yy[UU]

        yy[kk:DD * UU] = yy[kk]
        
        if bDebugsbs:
            print('yy[:{:d}] = {:3.3f} '.format( DD * UU - 1, yy[DD * UU - 1] ))
       
        # vector para filtrar muestras
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)

        # inicio de la recursión
        for kk in range(DD * UU, (DD * UU) + UU ):
    
            ii = kk-1
            # Calcula la salida según la ecuación recursiva
            yy[ii] = np.sum(xx[kk-(DD * UU):kk] * hh_u)
    
            if bDebugsbs:
                # # debug muestra a muestra
                # print(' yy[{:d}] = {:3.3f} '.format( ii, yy[ii]))
    
                print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( ii, kk-(DD * UU), ii ))
                xx_aux = xx[kk-(DD * UU):kk] * hh_u
                strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
                strAux = ' + '.join(strAux)
                strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
                print( strAux )

        # end_val = (DD * UU)
           
        # for kk in range(kk+1, end_val):
    
        #     # Calcula la salida según la ecuación recursiva
        #     yy[kk] = xx[kk] \
        #               - xx_ci[kk] \
        #               + yy[(kk - UU)]
    
        #     if bDebugsbs:
        #         dd = kk + kk_offset
        #         if kk <= (2*DD * UU):
        #             print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx_ci[{:d}] ({:3.3f}) + yy[{:d}] ({:3.3f})'.format( dd, yy[kk], dd, xx[kk], kk, xx_ci[kk], dd - UU, yy[kk - UU] ))

    else:
        # para todos los bloques restantes salvo el primero
           
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy_ci[kk]
    
            if bDebugsbs:
                
                dd = kk + kk_offset
                if kk <= (2*DD * UU):
            
                    print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx_ci[{:d}] ({:3.3f}) + yy_ci[{:d}] ({:3.3f})'.format( dd, yy[kk], dd, xx[kk], kk, xx_ci[kk], kk, yy_ci[kk] ))
                    print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx[{:d}] ({:3.3f}) + yy[{:d}] ({:3.3f})'.format( dd, yy[kk], kk, xx[kk], primera_xx_ci_idx+kk, xx_ci[kk], primera_yy_ci_idx+kk, yy_ci[kk] ))
        
        for kk in range(UU, DD * UU):

            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - xx_ci[kk] \
                      + yy[(kk - UU)]
    
            if bDebugsbs:
                dd = kk + kk_offset
                if kk <= (2*DD * UU):
                    print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx_ci[{:d}] ({:3.3f}) + yy[{:d}] ({:3.3f})'.format( dd, yy[kk], dd, xx[kk], kk, xx_ci[kk], dd - UU, yy[kk - UU] ))
    
        #
        kk += 1
    
    
    # for kk in range(NN):
    for kk in range(kk, NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk] = xx[kk]  \
                  - xx[kk - DD * UU] \
                  + yy[kk - UU]

        if bDebugsbs:
            dd = kk + kk_offset
            if kk <= (2*DD * UU) or \
               kk >= (NN - (DD * UU)):
                print(' yy[{:d} ({:d} + {:d})] = {:3.3f} = xx[{:d}  ({:d} + {:d})] ({:3.3f})  - xx[{:d} ({:d} + {:d} - {:d} * {:d})] ({:3.3f}) + yy[{:d} ({:d} + {:d} - {:d})] ({:3.3f})'.format( kk + kk_offset, kk, kk_offset, yy[kk], kk + kk_offset, kk, kk_offset, xx[kk], dd - DD * UU, kk, kk_offset, DD, UU, xx[ (kk - DD * UU)], dd - UU, kk, kk_offset, UU, yy[(kk - UU)]))
    
    # calculo las condiciones iniciales del siguiente bloque
    xx_ci = xx[primera_xx_ci_idx:]
    yy_ci = yy[primera_yy_ci_idx:]

    # escalo y devuelvo
    return( (yy.copy()/DD, xx_ci.copy(), yy_ci.copy()) )

## Implementación Python para RT 
def MA_st_rt_calc_ci( xx, DD, UU ):

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

## Implementación Python
def one_MA_stage( xx, DD, UU):
    
    NN = xx.shape[0]
    # buffer_size = DD * UU

    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)

    # resultaron ser importante las condiciones iniciales
    yy = np.zeros_like(xx)
    # yy = np.ones_like(xx) * xx[0] * DD * UU

    # condiciones iniciales
    yy[0:(DD * UU)] = np.sum(xx[0:(DD * UU)] * hh_u)

    kk = DD * UU - 1 
    if bDebugsbs:
        
        print('MA recursivo')
        print('------------')
        
        # # debug muestra a muestra
        print('... yy[{:d}] = {:3.3f} '.format( kk, yy[kk] ))

    # inicio de la recursión
    for kk in range(DD * UU, (DD * UU) + UU ):

        ii = kk-1
        # Calcula la salida según la ecuación recursiva
        yy[ii] = np.sum(xx[kk-(DD * UU):kk] * hh_u)

        if bDebugsbs:
            # # debug muestra a muestra
            # print(' yy[{:d}] = {:3.3f} '.format( ii, yy[ii]))

            print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( ii, kk-(DD * UU), ii ))
            xx_aux = xx[kk-(DD * UU):kk] * hh_u
            strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
            strAux = ' + '.join(strAux)
            strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
            print( strAux )


    # for kk in range(NN):
    for kk in range(kk, NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk] = xx[kk]  \
                  - xx[ (kk - DD * UU) % NN] \
                  + yy[(kk - UU) % NN]

        # # debug muestra a muestra
        if bDebugsbs:
            if (kk % block_s) <= (2*DD * UU) or \
               (kk % block_s) >= (block_s - (DD * UU)):
            
            # if (kk % NN) <= (2*DD * UU):
                print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx[{:d}] ({:3.3f}) + yy[{:d}] ({:3.3f})'.format( kk, yy[kk], kk, xx[kk], (kk - DD * UU) % NN, xx[ (kk - DD * UU) % NN], (kk - UU) % NN, yy[(kk - UU) % NN]))
                # print('    kk = {:d}         = {:d}    kk-UU = {:d}  '.format( , (kk - DD * UU) % NN, (kk - UU) % NN) )
            
        # print(' yy[kk] : {:3.3f} = xx[kk] : {:3.3f}  - xx[kk-DD*UU] : {:3.3f} + yy[kk-UU] : {:3.3f}'.format( yy[kk], xx[kk], xx[ (kk - DD * UU) % NN], yy[(kk - UU) % NN]))
        # print('    kk = {:d}        kk-DD*UU = {:d}    kk-UU = {:d}  '.format( kk, (kk - DD * UU) % NN, (kk - UU) % NN) )
    
    # escalo y devuelvo
    return( yy )

def Tdc_seq_removal_dbg( xx, DD = 16, UU = 2, MA_stages = 2 ):

    NN = xx.shape[0]

    # A partir de aquí diferentes formas de estimar el valor medio

    #################
    ## MA clasica
    #################

    if not bDbgMAcl:

        yy_c = np.zeros_like(xx)
        yyy_c = [yy_c.copy()]
        
    else:

        
        if bDebugsbs:
            
            sys.stdout = archivos['ma_cl']
    
            print('MA clasico')
            print('----------')
        
        # crucial escalar luego de calcular la salida de un MA
        yy_c = one_MA_stage_clasica( xx, DD, UU)/(DD)
        yyy_c = [yy_c.copy()]
        
        # cascadeamos MA_stages-1 más
        for ii in range(1, MA_stages):
                
            if bDebugsbs:
                print('MA_stage {:d}'.format(ii+1))
                print('------------')
                
            yy_c = one_MA_stage_clasica( yy_c, DD, UU)/(DD)
            yyy_c += [yy_c.copy()]

    #################
    ## MA recursivo
    #################


    if not bDbgMArec:

        yy = np.zeros_like(xx)
        yyy = [yy.copy()]
        
    else:
            

    
        if bDebugsbs:
            sys.stdout = archivos['ma_rec']
            
            print('MA recursivo')
            print('------------')
        
        # crucial escalar luego de calcular la salida de un MA
        yy = one_MA_stage( xx, DD, UU)/DD
        yyy = [yy.copy()]
        
        # cascadeamos MA_stages-1 más
        for ii in range(1, MA_stages):
                
            if bDebugsbs:
                print('MA_stage {:d}'.format(ii+1))
                print('------------')
                
            # crucial escalar luego de calcular la salida de un MA
            yy = one_MA_stage( yy, DD, UU)/(DD)
            yyy += [yy.copy()]
    

    #################
    ## MA rec RT
    #################


    if not bDbgMART:

        yy_rt = np.zeros_like(xx)
        yyy_rt = [yy_rt.copy()]
        
    else:
    
        if bDebugsbs:
    
            sys.stdout = archivos['ma_rt']
    
            print('MA rec RT')
            print('---------')
    
        # se calculan condiciones iniciales
        xx_ci, yy_ci = MA_st_rt_calc_ci( xx, DD, UU )
    
        yy_rt = np.zeros_like(xx)
    
        
        if bDebugsbs:
            print('Primer bloque')
            print('----------------')
        
        for jj in range(0, NN, block_s):
        
            yy_aux, xx_ci, yy_ci = MA_st_rt_dbg( xx[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
    
            yy_rt[jj:jj+block_s] = yy_aux
    
        yyy_rt = [yy_rt.copy()]
    
        # cascadeamos MA_stages-1 más
        for ii in range(1, MA_stages):
    
            if bDebugsbs:
                print('MA_stage {:d} RT'.format(ii+1))
                print('----------------')
            
            # se calculan condiciones iniciales
            xx_ci, yy_ci = MA_st_rt_calc_ci( yy_rt, DD, UU )
            
            for jj in range(0, NN, block_s):
            
                yy_aux, xx_ci, yy_ci = MA_st_rt_dbg( yy_rt[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
            
                yy_rt[jj:jj+block_s] = yy_aux
                
                
            # crucial escalar luego de calcular la salida de un MA
            yyy_rt += [yy_rt.copy()]
            

    #################
    # demora y resta
    
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU) )
    yy = xx_aux - yy
    yy_rt = xx_aux - yy_rt
    yy_c = xx_aux - yy_c
    # return( ( yy, yy_c, yyy, yyy_c  ) )
    return( ( yy, yy_c, yy_rt, yyy, yyy_rt, yyy_c ) )


#%% Parte numérica 

# Habilitar debug muestra a muestra
# bDebugsbs = True
bDebugsbs = False

# Habilitar debug comparación contra promedio clásico
bDbgMAcl = True
# bDbgMAcl = False

# Habilitar debug comparación contra promedio recursivo
# bDbgMArec = True
bDbgMArec = False

# Habilitar debug comparación contra promedio recursivo
# bDbgMART = True
bDbgMART = False


fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2

NN = 2**13
w_rad  = np.append(np.logspace(-2, 0.8, NN//4), np.logspace(0.9, 1.6, NN//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, NN//2, endpoint=True) ) / nyq_frec * np.pi

dc_val = 1

# respuesta al impulso: mala idea, solo tendremos soporte para DD**UU muestras.
# xx = np.zeros(NN)
# xx[0] = 1

dd = 16
uu = 4
ma_st = 2

# bloque de memoria para la implementación RT
# block_s = NN//2
block_s = NN


import scipy.io as sio

mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten().astype(np.float64)
cant_muestras = len(ecg_one_lead)

# xx = ecg_one_lead

# resp al ruido incorrelado
# xx = np.random.randn(NN) * dc_val/dd
# xx -=  np.mean(xx)
# xx /=  np.std(xx)
# xx +=  dc_val

# resp al impulso
# xx = np.zeros(NN)
# xx[2*uu*dd] = 1.0



if bDebugsbs:
    # Guarda la salida estándar actual
    stdout_original = sys.stdout
    
    # Especifica los nombres de los archivos de texto en los que deseas redirigir la salida
    nombres_archivos = [ ('ma_cl',"ma_clasico.txt"), \
                         ('ma_rec',"ma_recursivo.txt"), \
                         ('ma_rt',"ma_rec_rt.txt")]
    
    # Abre los archivos en modo de escritura
    archivos = {identificacion: open(nombre, 'w') for identificacion, nombre in nombres_archivos}    

try:


    # la versión clásica del Mov avg la tenía como referencia para depurar
    # la nueva 
    yy, yy_c, yy_rt, yyy, yyy_rt, yyy_c = Tdc_seq_removal_dbg( xx, DD = dd, UU = uu, MA_stages = ma_st )

finally:
    
    # Cierra todos los archivos y restaura la salida estándar original
    if bDebugsbs:
        
        for archivo in archivos.values():
            archivo.close()
        
        sys.stdout = stdout_original
        print('Archivos cerrados ...')

# señales intermedias de los MA
yyy = np.vstack(yyy).transpose()
yyy_c = np.vstack(yyy_c).transpose()
yyy_rt = np.vstack(yyy_rt).transpose()

## pruebitas

# yyy_rt[]xxxxxxxxxxxxxxxxxxxx

print(' xx = {:3.3f} +/- {:3.3f}  --  yy = {:3.3f} +/- {:3.3f} '.format( np.mean(xx), np.std(xx),  np.mean(yy), np.std(yy)))

# np.mean(yyy, auxis = 0)

print('MA recursivo')
[ print(' yyy = {:3.3f} +/- {:3.3f}'.format( np.mean(yyy[:,ii]), np.std(yyy[:,ii]))) for ii in range(yyy.shape[1])]

print('MA clásico')
[ print(' yyy = {:3.3f} +/- {:3.3f}'.format( np.mean(yyy_c[:,ii]), np.std(yyy_c[:,ii]))) for ii in range(yyy_c.shape[1])]

print('MA RT')
[ print(' yyy = {:3.3f} +/- {:3.3f}'.format( np.mean(yyy_rt[:,ii]), np.std(yyy_rt[:,ii]))) for ii in range(yyy_rt.shape[1])]

# nps = NN//2**2
# ff, psd_xx = sig.welch(xx, fs=2, nperseg=nps, detrend=False)
# ff, psd_yy = sig.welch(yy, fs=2, nperseg=nps, detrend=False)

# psd_xx = blackman_tukey( xx, NN//2**3 )
# psd_yy = blackman_tukey( yy, NN//2**3 )

# oversampling
# yy_u = np.zeros(NN*uu)
# yy_u[::uu] = yy
# yy_c_u = np.zeros(NN*uu)
# yy_c_u[::uu] = yy_c
# yyy_u = np.zeros((NN*uu, yyy.shape[1]))
# yyy_u[::uu,:] = yyy
# yyy_c_u = np.zeros((NN*uu, yyy_c.shape[1]))
# yyy_c_u[::uu,:] = yyy_c


psd_xx = 1/NN*np.abs(np.fft.fft(xx, axis=0))
psd_yy = 1/NN*np.abs(np.fft.fft(yy, axis=0))
psd_yy_c = 1/NN*np.abs(np.fft.fft(yy_c, axis=0))
psd_yy_rt = 1/NN*np.abs(np.fft.fft(yy_rt, axis=0))

# psd_xx_u = 1/NN*np.abs(np.fft.fft(xx_u, axis=0))
# psd_yy_u = (1/NN*np.abs(np.fft.fft(yy_u, axis=0)))
# psd_yy_c_u = (1/NN*np.abs(np.fft.fft(yy_c_u, axis=0)))

# psd_yyy = (1/NN*np.abs(np.fft.fft(yyy, axis=0)))
# psd_yyy_c = (1/NN*np.abs(np.fft.fft(yyy_c, axis=0)))

# psd_yyy_u = (1/NN*np.abs(np.fft.fft(yyy_u, axis=0)))
# psd_yyy_c_u = (1/NN*np.abs(np.fft.fft(yyy_c_u, axis=0)))

ff = np.arange(start=0, stop=fs/2, step = fs/NN)
# oversampleada
# ff_u = np.arange(start=0, stop=fs/2, step = fs/NN/uu)

psd_xx = psd_xx[:ff.shape[0]]
psd_yy = psd_yy[:ff.shape[0]]
psd_yy_c = psd_yy_c[:ff.shape[0]]
psd_yy_rt = psd_yy_rt[:ff.shape[0]]

# psd_xx_u = psd_xx_u[:ff_u.shape[0]]
# psd_yy_u = psd_yy_u[:ff_u.shape[0]]
# psd_yy_c_u = psd_yy_c_u[:ff_u.shape[0]]
# psd_yyy = psd_yyy[:ff.shape[0],:]
# psd_yyy_c = psd_yyy_c[:ff.shape[0],:]
# psd_yyy_u = psd_yyy_u[:ff_u.shape[0],:]
# psd_yyy_c_u = psd_yyy_c_u[:ff_u.shape[0],:]

plt.figure(1)
plt.clf()

if bDbgMArec:
    plt.plot(ff, 20*np.log10(psd_yy/psd_xx), ':^', label= 'Tdc_rec', alpha=0.5)

plt.plot(ff, 20*np.log10(psd_yy_rt/psd_xx), ':o', label= 'Tdc_rt', alpha=0.5)

if bDbgMAcl:
    plt.plot(ff, 20*np.log10(psd_yy_c/psd_xx), ':v', label= 'Tdc_cl', alpha=0.5)
    
# plt.plot(ff_u, 20*np.log10(psd_yy_c_u/psd_xx_u), label= 'Tdc_cl')
# plt.plot(ff_u, 20*np.log10(psd_yy_u/psd_xx_u), label= 'Tdc')
# plt.plot(ff_u, 20*np.log10(psd_xx_u), label= 'xx')
# plt.plot(ff, 20*np.log10(psd_yyy), label='MA rec')
# plt.plot(ff, 20*np.log10(psd_yyy_c), label='MA cl')
# plt.plot(ff_u, 20*np.log10(psd_yyy_u), label='MA rec')
# plt.plot(ff_u, 20*np.log10(psd_yyy_c_u), label='MA cl')
# plt.plot(ff_u, 20*np.log10(psd_yy_u), label= 'yy')
plt.legend()
# plt.axis([-10, 510, -100, 0 ])

# secuencia temporal para validar respecto al promedio clásico
plt.figure(2)
plt.clf()
plt.plot(xx, label= 'xx')

if bDbgMArec:
    plt.plot(yyy, ':^', label='MA rec', alpha=0.2, markersize=10)
    
plt.plot(yyy_rt, ':o', label='MA rt', alpha=0.5, markersize=5)

if bDbgMAcl:
    plt.plot(yyy_c, ':v', label='MA cl', alpha=0.2, markersize=15)
    
plt.plot(yy, label= 'yy', alpha=0.5)
plt.plot(yy_rt, label= 'yy_rt', alpha=0.5)
plt.plot(yy_c, label= 'yy_cl', alpha=0.5)
# plt.plot(ff, 20*np.log10(psd_yy/psd_xx))
# plt.axis([-10, 500, -100, 0 ])
plt.legend()

#%% Pruebas con ECG real

import matplotlib.pyplot as plt
import numpy as np

## Implementación Cython
import recursive_fir_filter as rt

import scipy.io as sio

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
ma_st = 4

# demora teórica del filtro de Rick
demora_rl = int((dd-1)/2*ma_st*uu)

if bDebugsbs:
    # Guarda la salida estándar actual
    stdout_original = sys.stdout
    
    # Especifica los nombres de los archivos de texto en los que deseas redirigir la salida
    nombres_archivos = [ ('ma_cl',"ma_clasico.txt"), \
                         ('ma_rec',"ma_recursivo.txt"), \
                         ('ma_rt',"ma_rec_rt.txt")]
    
    # Abre los archivos en modo de escritura
    archivos = {identificacion: open(nombre, 'w') for identificacion, nombre in nombres_archivos}    

try:

    ECG_f_rl, ECG_f_rl_c, ECG_f_rl_rt, yyy, yyy_rt, yyy_c  = Tdc_seq_removal_dbg( ecg_one_lead, DD = dd, UU = uu, MA_stages = ma_st )

finally:
    
    # Cierra todos los archivos y restaura la salida estándar original
    if bDebugsbs:
        
        for archivo in archivos.values():
            archivo.close()
        
        sys.stdout = stdout_original
        print('Archivos cerrados ...')



if bDebugsbs:

    # señales intermedias de los MA
    yyy = np.vstack(yyy).transpose()
    yyy_c = np.vstack(yyy_c).transpose()
    yyy_rt = np.vstack(yyy_rt).transpose()

    
    regs_interes = ( 
            
            np.array([0, 10000]) , # minutos a muestras
            np.array([-2000, 2000]) + block_s, # minutos a muestras
            )
    
    for ii in regs_interes:
        
        # intervalo limitado de 0 a cant_muestras
        zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='int')
        
        
        # secuencia temporal para validar respecto al promedio clásico
        plt.figure()
        plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
        
        if bDbgMArec:
            plt.plot(zoom_region, yyy[zoom_region+demora_rl,:], ':^', label='MA rec', alpha=0.2, markersize=10)
            
        if bDbgMART:
            plt.plot(zoom_region, yyy_rt[zoom_region+demora_rl,:], ':+', label='MA rt', alpha=0.5, markersize=15)
        
        if bDbgMAcl:
            plt.plot(zoom_region, yyy_c[zoom_region+demora_rl,:], ':v', label='MA cl', alpha=0.2, markersize=5)
    
           
        if bDbgMArec:
            plt.plot(zoom_region-demora_rl, ECG_f_rl[zoom_region], '-^', label= 'yy_rec', alpha=0.5, markersize=10)
        
        if bDbgMART:
            plt.plot(zoom_region-demora_rl, ECG_f_rl_rt[zoom_region], '-+', label= 'yy_rt', alpha=0.5, markersize=15)

        if bDbgMAcl:
            plt.plot(zoom_region-demora_rl, ECG_f_rl_c[zoom_region], '-v', label= 'yy_cl', alpha=0.5, markersize=5)
            
            
        # plt.plot(ff, 20*np.log10(psd_yy/psd_xx))
        # plt.axis([-10, 500, -100, 0 ])
        plt.legend()
    
else:

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
        if bDbgMArec:
            plt.plot(zoom_region, ECG_f_rl[zoom_region+demora_rl], ':^', alpha=0.5, label='rec')

        if bDbgMAcl:
            plt.plot(zoom_region, ECG_f_rl_c[zoom_region+demora_rl], ':v', alpha=0.5, label='CL')
        
        if bDbgMART:
            plt.plot(zoom_region, ECG_f_rl_rt[zoom_region+demora_rl], ':o', alpha=0.5, label='RT')
        
        plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
        plt.ylabel('Adimensional')
        plt.xlabel('Muestras (#)')
        
        axes_hdl = plt.gca()
        axes_hdl.legend()
                
        plt.show()
    

    
