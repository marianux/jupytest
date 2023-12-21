#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ej2 TP5: Filtros digitales

Created on Wed Aug 18 17:56:57 2021

@author: mariano
"""

import sympy as sp
# from pytc2.sistemas_lineales import plot_plantilla, simplify_n_monic
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


#%% Resolución simbólica

z = sp.symbols('z', complex=True)
D = sp.symbols('D', real=True, positive=True)
k = sp.symbols('k', integer=True, positive=True)

# promediador clásico
Tma_clasico = 1/D * sp.Sum(z**(-k), (k, 0, D-1))

# moving average
Tma = 1/D * (1-z**(-D))/(1-z**(-1))

# delay line of (D-1)/2
Tdl =  z**(-(D-1)/2)

num, den = (Tdl - Tma).as_numer_denom()

num = (sp.expand(num/(D*z**(D+1)))).powsimp()
den = (sp.expand(den/(D*z**(D+1)))).powsimp()

Tdc_removal = num/den

# display(Tdc_removal)

# Según Rick Lyons, este sistema sería muy bueno para implementarse
# con D múltiplo de 2**NN, dado que el escalado por D sería simplemente 
# una rotación a nivel de bits de NN veces a la derecha, y su implementación
# no necesitaría de multiplicaciones. Sin embargo esta elección impone un 
# retardo no entero. Por esta razón se opta por poner dos (incluso cuatro) 
# sistemas idénticos en cascada.

# Probamos primero con 2 moving average

Tdc_removal_2 = z**-(D-1) - Tma**2

# emprolijamos la expresion a mano
num, den = Tdc_removal_2.as_numer_denom()
num = (sp.expand(num/(D**2*z**(2*D+2))).powsimp())
den = (sp.expand(den/(D**2*z**(2*D+2))).powsimp())


Tdc_removal_2 = num/den

def transf_s_2ba( T_s ):
    
    num, den = sp.fraction(T_s)
    
    bb = np.array(num.as_poly(z**-1).all_coeffs(), dtype=float)
    aa = np.array(den.as_poly(z**-1).all_coeffs(), dtype=float)
    
    return( (bb,aa) )


# display(Tdc_removal_2)

# Ahora con 4 moving average

Tdc_removal_4 = z**-(2*D-2) - Tma**4

# emprolijamos la expresion
num, den = Tdc_removal_4.as_numer_denom()
num = (sp.expand(num/(D**4*z**(4*D+4)))).powsimp()
den = (sp.expand(den/(D**4*z**(4*D+4)))).powsimp()

Tdc_removal_4 = num/den

# display(Tdc_removal_4)

#%% Parte numérica 

fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2

NN = 2**12
w_rad  = np.append(np.logspace(-2, 0.8, NN//4), np.logspace(0.9, 1.6, NN//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, NN//2, endpoint=True) ) / nyq_frec * np.pi


def group_delay( freq, phase):
    
    dphase = -np.diff(np.unwrap(phase, period = 19/10* np.pi ))
    # dphase = -np.diff(phase)
    # # corregir saltos de fase
    # bAux = dphase > np.pi
    # dphase[bAux] = np.amin( np.hstack([dphase[bAux], dphase[bAux]-np.pi]), axis = 1 )
    groupDelay = dphase/np.diff(freq)
    
    return(np.append(groupDelay, groupDelay[-1]))
    


def Sym_freq_response(HH, zz, ww):
    
    # w = sp.Symbol('w', real=True)
    # HH_jw = HH.subs({z:1*sp.exp_polar(sp.I*w)})
    
    H_numeric = sp.lambdify(zz, HH, modules=['numpy'])
    
    z_vals = np.exp(1j * ww)  # Rango de frecuencias angulares
    
    # Evalúa la función numérica en el rango de frecuencias angulares
    magnitude_response = np.abs(H_numeric(z_vals))
    phase_response = np.angle(H_numeric(z_vals))  # La fase se devuelve en grados

    return((magnitude_response, phase_response))


def plt_freq_resp(title, magnitude_response, phase_response, w_rad, fs = 2):
    
    ww = w_rad / np.pi * fs/2

    # Grafica la respuesta en frecuencia de módulo
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(ww, 20 * np.log10(magnitude_response))
    
    plt.title('Respuesta en Frecuencia de Módulo: ' + title)
    plt.xlabel('Frecuencia Angular (w)')
    plt.ylabel('|H(jw)| (dB)')

    plt.axis([0, 100, -60, 5 ]);
    
    gd = group_delay( w_rad, phase_response)
    mgd = np.nanmedian(gd)
    
    # Grafica la respuesta en frecuencia de retardo
    plt.subplot(2, 1, 2)
    plt.plot(ww, gd)
    plt.title('Respuesta de retardo')
    plt.xlabel('Frecuencia Angular (w)')
    plt.ylabel('Retardo (#)')
    plt.axis([0, 100, mgd-5, mgd+5 ]);
    
    plt.tight_layout()
    plt.show()
    

def blackman_tukey(x,  M = None):    
    
    N = len(x)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    return Px;


# Del análisis simbólico
DD = [3, 5, 11]
UU = 20
  

# Grafica la respuesta en frecuencia de módulo

plt.figure(1)
plt.clf()

ww = w_rad / np.pi * fs / 2

for ddd in DD:
    
    # Cálculo de demoras para mov. avg
    demora_ma = int(UU*(ddd-1))

    Tma_ddd = Tma.subs({z:z**UU, D:ddd})

    mod_Tma_ddd, pha_Tma_ddd = Sym_freq_response(Tma_ddd, z, w_rad )

    plt.plot(ww, 20 * np.log10(mod_Tma_ddd), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_ma) )

plt.title('Respuesta en Frecuencia de Módulo: RL-MovAvgRec-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 500, -80, 1 ]);

plt.figure(2)
plt.clf()

ww = w_rad / np.pi * fs / 2

for ddd in DD:
    
    # Cálculo de demoras para mov. avg
    demora_ma = int(UU*(ddd-1))

    Tma_c_ddd = Tma_clasico.subs({z:z**UU, D:ddd})
    
    Tma_c_ddd = Tma_c_ddd.doit()

    mod_Tma_c_ddd, pha_Tma_c_ddd = Sym_freq_response(Tma_c_ddd, z, w_rad )

    plt.plot(ww, 20 * np.log10(mod_Tma_c_ddd), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_ma) )

plt.title('Respuesta en Frecuencia de Módulo: RL-MovAvgClasico-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 500, -80, 1 ]);



plt.figure(3)
plt.clf()


for ddd in DD:
    
    # Cálculo de demoras para 2 mov. avg
    demora_rl2 = int(UU*(ddd-1))

    Tdcr_2 = Tdc_removal_2.subs({z:z**UU, D:ddd})

    mod_Tdcr_2, pha_Tdcr_2 = Sym_freq_response(Tdcr_2, z, w_rad )

    # plt_freq_resp('FIR-RL-2MA-D{:d}-OverS:{:d}'.format(DD, UU), mod_Tdcr_2, pha_Tdcr_2, w_rad, fs = fs)
    plt.plot(ww, 20 * np.log10(mod_Tdcr_2), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_rl2) )

plt.title('Respuesta en Frecuencia de Módulo: RL-2MA-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 100, -1, 0.5 ]);

plt.figure(4)
plt.clf()

for ddd in DD:

    # Cálculo de demoras para 4 mov. avg
    demora_rl4 = int(2*UU*(ddd-1))
    
    Tdcr_4 = Tdc_removal_4.subs({z:z**UU, D:ddd})
    
    mod_Tdcr_4, pha_Tdcr_4 = Sym_freq_response(Tdcr_4, z, w_rad )
    
    # plt_freq_resp('FIR-RL-4MA-D{:d}-OverS:{:d}'.format(DD, UU), mod_Tdcr_4, pha_Tdcr_4, w_rad, fs = fs)
    
    # plt.subplot(2, 1, 1)
    plt.plot(ww, 20 * np.log10(mod_Tdcr_4), label = 'D:{:d} (#) - GD:{:3.1f} (#)'.format(ddd, demora_rl4) )

plt.title('Respuesta en Frecuencia de Módulo: RL-4MA-OverS:{:d}'.format(UU))
plt.xlabel('Frecuencia Angular (w)')
plt.ylabel('|H(jw)| (dB)')
plt.legend()
plt.axis([0, 100, -1, 0.5 ]);

plt.tight_layout()
plt.show()





# La respuesta de fase es tan grande que se dificulta calcular y visualizar.
# ver variable demora_rl2
# 
# gd2 = group_delay( w_rad, pha_Tdcr_2)
# gd4 = group_delay( w_rad, pha_Tdcr_4)
# mgd2 = np.nanmedian(gd2)
# mgd4 = np.nanmedian(gd4)

# plt.figure(2)
# # Grafica la respuesta en frecuencia de retardo
# # plt.subplot(2, 1, 2)
# plt.plot(ww, gd2, label = 'FIR-RL-2MA-D{:d}-OverS:{:d}'.format(DD, UU))
# plt.plot(ww, gd4, label = 'FIR-RL-4MA-D{:d}-OverS:{:d}'.format(DD, UU))

# plt.title('Respuesta de retardo')
# plt.xlabel('Frecuencia Angular (w)')
# plt.ylabel('Retardo (#)')
# plt.axis([0, 100, np.min([mgd2,mgd4])-5, np.max([mgd2,mgd4])+5 ]);

# plt.tight_layout()
# plt.show()


# coeficientes
# bb2, aa2 = transf_s_2ba( Tdcr_2)

# bb4, aa4 = transf_s_2ba( Tdcr_4)

#%% Implementación via Cython

# from recursive_fir_filter import filter_sequence


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
    # kk = DD * UU
    # print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( kk-1, kk-(DD * UU), kk-1 ))
    # xx_aux = xx[kk-(DD * UU):kk] * hh_u
    # strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
    # strAux = ' + '.join(strAux)
    # strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
    # print( strAux )
    
    for kk in range(DD * UU,NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk-1] = np.sum(xx[kk-(DD * UU):kk] * hh_u)

        # # debug muestra a muestra
        # if kk <= (3*DD * UU):
        #     print(' yy[{:d}] = np.sum(xx[{:d}:{:d}])'.format( kk-1, kk-(DD * UU), kk-1 ))
        #     xx_aux = xx[kk-(DD * UU):kk] * hh_u
        #     strAux = ['{:3.3f}'.format(xx_aux[ii]) for ii in range(xx_aux.shape[0]) ]
        #     strAux = ' + '.join(strAux)
        #     strAux = ' {:3.3f} = '.format(yy[kk-1]) + strAux
        #     print( strAux )
    
    # escalo y devuelvo
    return( yy )

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
    # # debug muestra a muestra
    # print('... yy[{:d}] = {:3.3f} '.format( kk, yy[kk] ))

    # inicio de la recursión
    for kk in range(DD * UU, (DD * UU) + UU ):

        ii = kk-1
        # Calcula la salida según la ecuación recursiva
        yy[ii] = np.sum(xx[kk-(DD * UU):kk] * hh_u)

        # # debug muestra a muestra
        # print(' yy[{:d}] = {:3.3f} '.format( ii, yy[ii]))

    # for kk in range(NN):
    for kk in range(kk, NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk] = xx[kk]  \
                  - xx[ (kk - DD * UU) % NN] \
                  + yy[(kk - UU) % NN]

        # # debug muestra a muestra
        # if kk <= (3*DD * UU):
        #     print(' yy[{:d}] = {:3.3f} = xx[{:d}] ({:3.3f})  - xx[{:d}] ({:3.3f}) + yy[{:d}] ({:3.3f})'.format( kk, yy[kk], kk, xx[kk], (kk - DD * UU) % NN, xx[ (kk - DD * UU) % NN], (kk - UU) % NN, yy[(kk - UU) % NN]))
        #     # print('    kk = {:d}         = {:d}    kk-UU = {:d}  '.format( , (kk - DD * UU) % NN, (kk - UU) % NN) )
            
        # print(' yy[kk] : {:3.3f} = xx[kk] : {:3.3f}  - xx[kk-DD*UU] : {:3.3f} + yy[kk-UU] : {:3.3f}'.format( yy[kk], xx[kk], xx[ (kk - DD * UU) % NN], yy[(kk - UU) % NN]))
        # print('    kk = {:d}        kk-DD*UU = {:d}    kk-UU = {:d}  '.format( kk, (kk - DD * UU) % NN, (kk - UU) % NN) )
    
    # escalo y devuelvo
    return( yy )


def Tdc_seq_removal( xx, DD = 16, UU = 2, MA_stages = 2 ):
    
    # crucial escalar luego de calcular la salida de un MA
    yy = one_MA_stage( xx, DD, UU)/(DD )
    yy_c = one_MA_stage_clasica( xx, DD, UU)/(DD)

    yyy = [yy]
    yyy_c = [yy_c]
    
    # cascadeamos MA_stages-1 más
    for ii in range(1, MA_stages):
        
        # crucial escalar luego de calcular la salida de un MA
        yy = one_MA_stage( yy, DD, UU)/(DD)
        yy_c = one_MA_stage_clasica( yy_c, DD, UU)/(DD)
        yyy += [yy]
        yyy_c += [yy_c]

    # yyy += yyy_c
    
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU) )
    yy = xx_aux - yy
    yy_c = xx_aux - yy_c
    return( ( yy, yy_c, yyy, yyy_c  ) )


dc_val = 1

# respuesta al impulso: mala idea, solo tendremos soporte para DD**UU muestras.
# xx = np.zeros(NN)
# xx[0] = 1

dd = 16
uu = 4
ma_st = 2

xx = np.random.randn(NN) * dc_val/dd
# xx -=  np.mean(xx)
# xx /=  np.std(xx)
xx +=  dc_val


xx_u = np.zeros(NN*uu)
xx_u[::uu] = xx

# xxx = xx
# xxx = xx_u

yy, yy_c, yyy, yyy_c = Tdc_seq_removal( xx, DD = dd, UU = uu, MA_stages = ma_st )


# señales intermedias de los MA
yyy = np.vstack(yyy).transpose()
yyy_c = np.vstack(yyy_c).transpose()

## pruebitas
# yyy1 = sig.lfilter(np.array([1, 1])*1/2, 1, xx)

print(' xx = {:3.3f} +/- {:3.3f}  --  yy = {:3.3f} +/- {:3.3f} '.format( np.mean(xx), np.std(xx),  np.mean(yy), np.std(yy)))

# np.mean(yyy, auxis = 0)

print('MA recursivo')
[ print(' yyy = {:3.3f} +/- {:3.3f}'.format( np.mean(yyy[:,ii]), np.std(yyy[:,ii]))) for ii in range(yyy.shape[1])]

print('MA clásico')
[ print(' yyy = {:3.3f} +/- {:3.3f}'.format( np.mean(yyy[:,ii]), np.std(yyy[:,ii]))) for ii in range(yyy.shape[1])]

# nps = NN//2**5
# ff, psd_xx = sig.welch(xx, fs=2, nperseg=nps, detrend=False)
# ff, psd_yy = sig.welch(yy, fs=2, nperseg=nps, detrend=False)

# psd_xx = blackman_tukey( xx, NN//2**3 )
# psd_yy = blackman_tukey( yy, NN//2**3 )

# oversampling
yy_u = np.zeros(NN*uu)
yy_u[::uu] = yy
yy_c_u = np.zeros(NN*uu)
yy_c_u[::uu] = yy_c
yyy_u = np.zeros((NN*uu, yyy.shape[1]))
yyy_u[::uu,:] = yyy
yyy_c_u = np.zeros((NN*uu, yyy_c.shape[1]))
yyy_c_u[::uu,:] = yyy_c

# psd_xx = 1/NN*np.abs(np.fft.fft(xxx, axis=0))
# psd_yy = (1/NN*np.abs(np.fft.fft(yy, axis=0)))

psd_xx_u = 1/NN*np.abs(np.fft.fft(xx_u, axis=0))
psd_yy_u = (1/NN*np.abs(np.fft.fft(yy_u, axis=0)))
psd_yy_c_u = (1/NN*np.abs(np.fft.fft(yy_c_u, axis=0)))

psd_yyy = (1/NN*np.abs(np.fft.fft(yyy, axis=0)))
psd_yyy_c = (1/NN*np.abs(np.fft.fft(yyy_c, axis=0)))

psd_yyy_u = (1/NN*np.abs(np.fft.fft(yyy_u, axis=0)))
psd_yyy_c_u = (1/NN*np.abs(np.fft.fft(yyy_c_u, axis=0)))

# ff = np.arange(start=0, stop=fs/2, step = fs/NN)
# oversampleada
ff_u = np.arange(start=0, stop=fs/2, step = fs/NN/uu)

# psd_xx = psd_xx[:ff.shape[0]]
# psd_yy = psd_yy[:ff.shape[0]]

psd_xx_u = psd_xx_u[:ff_u.shape[0]]
psd_yy_u = psd_yy_u[:ff_u.shape[0]]
psd_yy_c_u = psd_yy_c_u[:ff_u.shape[0]]
# psd_yyy = psd_yyy[:ff.shape[0],:]
# psd_yyy_c = psd_yyy_c[:ff.shape[0],:]
psd_yyy_u = psd_yyy_u[:ff_u.shape[0],:]
psd_yyy_c_u = psd_yyy_c_u[:ff_u.shape[0],:]

plt.figure(1)
plt.clf()
plt.plot(ff_u, 20*np.log10(psd_yy_u/psd_xx_u), label= 'Tdc')
plt.plot(ff_u, 20*np.log10(psd_yy_c_u/psd_xx_u), label= 'Tdc_cl')
# plt.plot(ff_u, 20*np.log10(psd_yy_u/psd_xx_u), label= 'Tdc')
# plt.plot(ff_u, 20*np.log10(psd_xx_u), label= 'xx')
# plt.plot(ff, 20*np.log10(psd_yyy), label='MA rec')
# plt.plot(ff, 20*np.log10(psd_yyy_c), label='MA cl')
# plt.plot(ff_u, 20*np.log10(psd_yyy_u), label='MA rec')
# plt.plot(ff_u, 20*np.log10(psd_yyy_c_u), label='MA cl')
# plt.plot(ff_u, 20*np.log10(psd_yy_u), label= 'yy')
plt.legend()
# plt.axis([-10, 510, -100, 0 ])

plt.figure(2)
plt.clf()
plt.plot(xx, label= 'xx')
plt.plot(yyy, label='MA rec')
plt.plot(yyy_c, label='MA cl')
plt.plot(yy, label= 'yy')
# plt.plot(ff, 20*np.log10(psd_yy/psd_xx))
# plt.axis([-10, 500, -100, 0 ])
plt.legend()

# Tdcr_2 = Tdc_removal_2.subs({z:z**uu, D:dd})
# # coeficientes
# bb2, aa2 = transf_s_2ba( Tdcr_2 )


# # fpw = w0*np.pi*fs/np.tan(np.pi/2*w0); 

# ## Rick Lyons ECG filter

# # demora_rl = int(uu*(dd-1))
# demora2_rl = (len(bb2)-1)/2
# demora2_rl = (len(bb4)-1)/2



# z,p,k = sig.tf2zpk(bb2, aa2)

# sos_rl2 = sig.tf2sos(bb2, aa2, pairing='nearest')
# sos_rl4 = sig.tf2sos(bb4, aa4, pairing='nearest')

# _, hh2_rl = sig.sosfreqz(sos_rl2, w_rad)
# _, hh4_rl = sig.sosfreqz(sos_rl4, w_rad)

# w = w_rad / np.pi * nyq_frec

# plt.close('all')

# plt.figure(1)
# plt.clf()

# plt.plot(w, 20 * np.log10(abs(hh2_rl)), label='FIR-RL-2-D{:d} orden:{:d}'.format(2, DD))
# plt.plot(w, 20 * np.log10(abs(hh4_rl)), label='FIR-RL-2-D{:d} orden:{:d}'.format(4, DD))
# # plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

# plt.title('FIR diseñado por métodos directos')
# plt.xlabel('Frequencia [Hz]')
# plt.ylabel('Modulo [dB]')
# plt.axis([0, 100, -60, 5 ]);
# plt.legend()

# plt.grid()

