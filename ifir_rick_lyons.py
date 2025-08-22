#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:22:44 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio

from pytc2.filtros_digitales import DC_PWL_removal_recursive_filter

def empirical_frequency_response(filter_obj, freq_points=1024, signal_length=8192, fs = 2):
    """
    Calcula la respuesta en frecuencia empírica excitando el filtro con ruido blanco
    
    Parámetros:
    filter_obj : Instancia de TFilter
    freq_points : Número de puntos en frecuencia a evaluar
    signal_length : Longitud de la señal de prueba
    
    Retorna:
    freqs : Frecuencias normalizadas (0 a π)
    response : Respuesta en frecuencia compleja
    """
    # Generar señal de prueba (ruido blanco)
    np.random.seed(42)  # Para reproducibilidad
    x = np.random.normal(1, 1, signal_length)
    
    # print(f'Freq Response')
    
    # Procesar señal a través del filtro
    y = filter_obj.process(x)
    
    # Eliminar transitorios iniciales
    discard = min(t_filter.samp_avg * t_filter.upsample, signal_length // 10)
    x = x[discard:]
    y = y[discard:]
    
    welch_avg = 10
    # Calcular densidad espectral cruzada y autoespectro
    f, Pxy = signal.csd(x, y, nperseg=signal_length//welch_avg, nfft=signal_length, fs = fs)
    f, Pxx = signal.welch(x, nperseg=signal_length//welch_avg, nfft=signal_length, fs = fs)
    
    # Respuesta en frecuencia estimada
    H_empirical = Pxy / Pxx
    
    return f, H_empirical
   
def theoretical_impulse_response(D, U=1, cant_ma = 2, length=100):
    """
    Calcula la respuesta al impulso teórica de T(z) = z^{-(D-1)U} - Tₘₐ²(z)
    
    Parámetros:
    D : int - Retardo base
    U : int - Factor de sobremuestreo
    length : int - Longitud de la respuesta a calcular
    
    Retorna:
    h : ndarray - Respuesta al impulso teórica
    """
    h = np.zeros(length)
    
    # Componente del retardo puro
    delay_pos =  int(cant_ma/2* (D - 1) * U)
    if delay_pos < length:
        h[delay_pos] = 1
    
    # Componente de Tₘₐ²(z)
    # Calculamos la respuesta al impulso de Tₘₐ² como la convolución de dos promedios móviles
    ma_response = np.zeros(length)
    ma_response[:D*U:U] = 1/(D)  # Respuesta de un Tₘₐ
    
    ma_out = ma_response
    
    # Convolucionamos para obtener Tₘₐ²
    for ii in range(1, cant_ma):
        ma_out = np.convolve(ma_response, ma_out)[:length]
    
    # Combinamos según T(z) = retardo - Tₘₐ²
    h_theoretical = h - ma_out
    
    return h_theoretical

def compare_impulse_responses(filter_obj, D, U, cant_ma, length=60):
    """
    Compara la respuesta al impulso teórica y empírica
    
    Parámetros:
    filter_obj : Instancia de TFilter
    D : int - Retardo usado en el filtro
    U : int - Sobremuestreo usado en el filtro
    length : int - Longitud de la respuesta a analizar
    """
    # Obtenemos ambas respuestas
    h_empirical = filter_obj.impulse_response(length)
    h_theoretical = theoretical_impulse_response(D, U, cant_ma, length)
    
    # Calculamos el error
    error = np.abs(h_empirical - h_theoretical)
    
    # Graficamos
    # Creamos la figura y los subplots con sharex=True
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # Graficamos en cada subplot
    # Subplot 1: Respuesta empírica
    ax1.stem(h_empirical, markerfmt='bo', linefmt='b-', basefmt=' ', label='Empírica')
    ax1.set_title(f'Comparación de Respuestas al Impulso (MA_avg={D}, MA_stages = {cant_ma}, Upsample={U} )')
    ax1.set_ylabel('Empírica')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Respuesta teórica
    ax2.stem(h_theoretical, markerfmt='ro', linefmt='r-', basefmt=' ', label='Teórica')
    ax2.set_ylabel('Teórica')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Error absoluto
    ax3.stem(error, markerfmt='go', linefmt='g-', basefmt=' ')
    ax3.set_ylabel('Error absoluto')
    ax3.set_xlabel('Muestras')
    ax3.grid(True, alpha=0.3)
    
    # Ajustamos el layout para evitar solapamientos
    plt.tight_layout()
    plt.show()
    
    # Métricas de error
    print(f"\nMétricas de error para D={D}, U={U}:")
    print(f"Error máximo: {np.max(error):.2e}")
    print(f"Error RMS: {np.sqrt(np.mean(error**2)):.2e}")
    print(f"Error relativo medio: {np.mean(np.abs(error)/np.abs(h_theoretical + 1e-12)):.2e}")
    

# %% Ejemplo de uso


if __name__ == "__main__":

    fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
    
    dd = 8
    uu = 1
    ma_st = 2
   
    bDebug = True
    # bDebug = False

    
    if bDebug:
    
        # Ejemplo de uso
                
        t_filter = DC_PWL_removal_recursive_filter(samp_avg = dd, upsample = uu, cant_ma = ma_st, batch = None, fpwl = 0.2 )
        # t_filter = TFilter(dd, uu, cant_ma = ma_st )
    
        freq_resp_length = 2**16
        # w_emp_fft, H_emp_fft = empirical_frequency_response(t_filter, fs = fs, signal_length = freq_resp_length)
        w_emp_fft, H_emp_fft =         t_filter.frequency_response(n_freq = freq_resp_length, bTeorica=False)
        w_theoretical, H_theoretical = t_filter.frequency_response(n_freq = freq_resp_length, bTeorica=True)
        w_theoretical = w_theoretical/np.pi*t_filter.fs/2
    
        
        # Gráficos comparativos
        plt.figure(figsize=(14, 8))
    
        plt.plot(w_theoretical, 20*np.log10(np.abs(H_theoretical)), 'b-', label='Teórica')
        plt.plot(w_emp_fft, 20*np.log10(np.abs(H_emp_fft)), 'r--', label='Empírica (FFT)')
        plt.title(f'Respuesta en Frecuencia - Magnitud (MA_avg={dd}, MA_stages = {ma_st}, Upsample={uu} )')
        plt.ylabel('Magnitud (dB)')
        plt.legend()
        plt.grid(True)
    
        compare_impulse_responses(t_filter, dd, uu, ma_st, length=2*ma_st*uu*dd)

    else:
        

        import wfdb 
        import os
        
        
        # Ruta al archivo y lectura
        carpeta = r'/home/mariano/Downloads/senialesvarias'
        # nombre = '00001_hr'
        # nombre = '00010_hr'
        # nombre = '4991'
        # nombre = '8760'
        # nombre = '64'
        # nombre = '113'           # VER CON LYONS
        nombre = '1_PR01_060803_1'
        # nombre = '3_PR01_060803_3'
        # nombre = '4_PR01_110803_1'
        # nombre = '5_PR01_110803_2'

        
        record = wfdb.rdrecord(os.path.join(carpeta, nombre))
        
        # Transponer la matriz: derivaciones en filas
        ecg_one_lead = record.p_signal  # shape = (n_derivaciones, n_muestras)
        fs = record.fs
        n_muestras = record.sig_len

        # uu = fs // 50

        # mat_struct = sio.loadmat('ecg.mat')
       # ecg_one_lead = mat_struct['ecg_lead']
        # ecg_one_lead = ecg_one_lead.flatten().astype(np.float64)
        cant_muestras = n_muestras

        # t_filter = TFilter(dd, uu, cant_ma = ma_st )
        # t_filter = DC_PWL_removal_recursive_filter(fs, samp_avg = dd, upsample = uu, cant_ma = ma_st, batch = int(np.ceil(cant_muestras/5)) )
        t_filter = DC_PWL_removal_recursive_filter(fs, samp_avg = dd, cant_ma = ma_st)
        
        ECG_f_rl_fin = t_filter.process(ecg_one_lead)

        demora_rl = t_filter.demora
        
        # Nota: se obtuvo esta performance en una notebook estandard con:
        # HP 255 15.6 inch G10 Notebook PC (A82ZVUA#ABA)
        # 8BA5
        # AMD Ryzen 7 7730U with Radeon Graphics 64KiB BIOS 512KiB L1 caché 4MiB L2 caché 16MiB L3 caché
        # 16GiB Memoria de sistema SODIMM DDR4 Síncrono Unbuffered (Unregistered) 3200 MHz (0,3 ns)
        # 1129116 muestras de ECG a fs = 1kHz
        # %timeit ECG_f_rl_fin = t_filter.process(ecg_one_lead)
        # 1.18 s ± 10.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        plt.close('all')
        
        regs_interes = ( 
                
                np.array([0, cant_muestras-1]), # minutos a muestras
                # np.array([0, 1]) *60*fs, # minutos a muestras
                # np.array([2, 2.2]) *60*fs, # minutos a muestras
                # np.array([5, 5.2]) *60*fs, # minutos a muestras
                # np.array([10, 10.2]) *60*fs, # minutos a muestras
                # np.array([12, 12.4]) *60*fs, # minutos a muestras
                # np.array([15, 15.2]) *60*fs, # minutos a muestras
                # np.array([18, 18.2]) *60*fs, # minutos a muestras
                # [4000, 5500], # muestras
                # [10e3, 11e3], # muestras
                )
        
        for ii in regs_interes:
            
            # intervalo limitado de 0 a cant_muestras
            zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras-1, ii[1]]), dtype='uint')
            

            for jj in range(record.n_sig):
            
                plt.figure()
                plt.plot(zoom_region, ecg_one_lead[zoom_region,jj], label='ECG', linewidth=2)
            
                # FIR con corrección de demora
                # plt.plot(zoom_region, ECG_f_rl_fin[zoom_region+demora_rl,jj], ':x', alpha=0.5, label='final')
                plt.plot(zoom_region, np.roll(ECG_f_rl_fin[zoom_region,jj], -demora_rl), label='RLfilt')
                
                plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
                plt.ylabel('Adimensional')
                plt.xlabel('Muestras (#)')
                
                axes_hdl = plt.gca()
                axes_hdl.legend()

                        
            plt.show()
        
                
    