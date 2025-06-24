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


class MA_hp_filter:
    
    def __init__(self, samp_avg = 16, upsample = 1):
        """
        Implementación recursiva de Tₘₐ(z) = (1 - z^{-D*U})/(D*(1 - z^{-U}))
        
        Parámetros:
        samp_avg : int - Muestras a promediar
        upsample : int - Factor de sobremuestreo (default=1)
        
        """
        self.samp_avg = samp_avg
        self.upsample = upsample
        self.effective_D = samp_avg * upsample
        
        # se asumen condiciones iniciales nulas
        self.yy_ci = np.zeros(upsample)
        self.xx_ci = np.zeros(self.effective_D)
        
        self.kk_offset = 0


    def reset(self):

        # se asumen condiciones iniciales nulas
        self.yy_ci = np.zeros(self.upsample)
        self.xx_ci = np.zeros(self.effective_D)


    def process(self, xx, xx_ci = None, yy_ci = None):
        
        NN = xx.shape[0]
    
        if xx_ci is None:
            xx_ci = self.xx_ci

        if yy_ci is None:
            yy_ci = self.yy_ci

    
        # resultaron ser importante las condiciones iniciales
        yy = np.zeros_like(xx)
        # yy = np.ones_like(xx) * xx[0] * self.effective_D
    
        # para todos los bloques restantes salvo el primero
           
        for kk in range(self.upsample):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - self.xx_ci[kk] \
                      + self.yy_ci[kk]
        
        for kk in range(self.upsample, self.effective_D):

            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk] \
                      - self.xx_ci[kk] \
                      + yy[(kk - self.upsample)]
    
        #
        kk += 1
        
        # for kk in range(NN):
        for kk in range(kk, NN):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk] = xx[kk]  \
                      - xx[kk - self.effective_D] \
                      + yy[kk - self.upsample]
        
        # calculo las condiciones iniciales del siguiente bloque
        xx_ci = xx[(NN - self.effective_D):]
        yy_ci = yy[(NN - self.upsample):]

        self.xx_ci = xx_ci 
        self.yy_ci = yy_ci
    
        # escalo y devuelvo
        return( (yy.copy()/self.samp_avg, xx_ci.copy(), yy_ci.copy()) )

    def impulse_response(self, length=100):
        """Genera la respuesta al impulso"""
        impulse = np.zeros(length)
        impulse[0] = 1
        
        self.reset()
        
        return self.process(impulse)
    
    def frequency_response(self, n_freq=1000):
        """Calcula la respuesta en frecuencia"""
        
        # Forma directa (no recursiva) para verificación
        b = np.zeros(self.effective_D + 1)
        b[0] = 1
        b[self.effective_D] = -1
        b = b / self.samp_avg
        
        a = np.zeros(self.upsample + 1)
        a[0] = 1
        a[self.upsample] = -1
        
        w, h = signal.freqz(b, a, worN=n_freq)
        return w, h


class ECGcomb:
    
    def __init__(self, samp_avg = 16, cant_ma = 2, upsample = 1, batch = None ):
        """
        Implementación completa de:
        T(z) = z^{-(D-1)*U} - Tₘₐ²(z)

        Implementación recursiva de Tₘₐ(z) = (1 - z^{-D*U})/(D*(1 - z^{-U}))
        
        Parámetros:
        samp_avg : int - Muestras a promediar
        upsample : int - Factor de sobremuestreo (default=1)
        
        batch : int - Bloque batch por bloques. Default: toda la señal.
        
        """
        self.samp_avg = samp_avg
        self.upsample = upsample
        self.cant_ma = (cant_ma//2) * 2
        self.samp_avgelay = int( self.cant_ma/2*(samp_avg - 1) * upsample )
        self.batch = batch

        t_ma = [MA_hp_filter(samp_avg = self.samp_avg, upsample = self.upsample)]

        for ii in range(1, self.cant_ma):
            
            t_ma += [MA_hp_filter(samp_avg = self.samp_avg, upsample = self.upsample)]
        
        self.t_ma = t_ma


    def reset(self):

        for ii in range(self.cant_ma):

            self.t_ma[ii].reset()
            
        
    def process(self, xx):

            
        yy = np.zeros_like(xx)
        
        NN = xx.shape[0]

        if self.batch is None:
            
            self.batch = NN
        
        # se procesa cada bloque por separado y se concatena la salida
        for jj in range(0, NN, self.batch):

            yy_aux,_,_ = self.t_ma[0].process(xx[jj:jj+self.batch])
    
            yy[jj:jj+self.batch] = yy_aux
    
        # cascadeamos MA_stages-1 más
        for ii in range(1, self.cant_ma):
    
            # se procesa cada bloque por separado y se concatena la salida
            for jj in range(0, NN, self.batch):
    
                yy_aux,_,_ = self.t_ma[ii].process(yy[jj:jj+self.batch] )
        
                yy[jj:jj+self.batch] = yy_aux


        # Aplicar retardo (D-1)*U
        delayed_x = np.roll(xx, self.samp_avgelay)
        delayed_x[:self.samp_avgelay] = 0
        
        return delayed_x - yy
    
    def impulse_response(self, length=100):
        
        impulse = np.zeros(length)
        impulse[0] = 1
        
        self.reset()
        
        return self.process(impulse)
    
    def frequency_response(self, n_freq=1000):
        
        w, h_ma_sq = self.t_ma[0].frequency_response(n_freq)
        h_delay = np.exp(-1j * w * self.samp_avgelay)
        
        return w, h_delay - h_ma_sq**self.cant_ma


class MovingAverageFilter:
    def __init__(self, D, U=1):
        """
        Implementación recursiva de Tₘₐ(z) = (1 - z^{-D*U})/(D*(1 - z^{-U}))
        
        Parámetros:
        D : int - Retardo base
        U : int - Factor de sobremuestreo (default=1)
        """
        self.samp_avg = D
        self.upsample = U
        self.effective_D = D * U
        
        # Buffers para implementación recursiva
        self.x_buffer = np.zeros(self.effective_D+1)
        self.y_buffer = np.zeros(U+1)
        self.y_prev = 0.0
        
    def process(self, x):
        
        y = np.zeros_like(x, dtype=float)
        self.x_buffer = np.zeros(self.effective_D+1)
        self.y_buffer = np.zeros(self.upsample+1)
        self.y_prev = 0.0
        
        for n in range(len(x)):
            # Actualizar buffer de entrada
            self.y_buffer = np.roll(self.y_buffer, 1)
            self.x_buffer = np.roll(self.x_buffer, 1)
            self.x_buffer[0] = x[n]

            # Calcular salida (forma recursiva)
            y[n] = (self.x_buffer[0] - self.x_buffer[self.effective_D]) + self.y_buffer[self.upsample]

            self.y_buffer[0] = y[n]
        
        return y / self.samp_avg

    def impulse_response(self, length=100):
        """Genera la respuesta al impulso"""
        impulse = np.zeros(length)
        impulse[0] = 1
        return self.process(impulse)
    
    def frequency_response(self, n_freq=1000):
        """Calcula la respuesta en frecuencia"""
        
        # Forma directa (no recursiva) para verificación
        b = np.zeros(self.effective_D + 1)
        b[0] = 1
        b[self.effective_D] = -1
        b = b / self.samp_avg
        
        a = np.zeros(self.upsample + 1)
        a[0] = 1
        a[self.upsample] = -1
        
        w, h = signal.freqz(b, a, worN=n_freq)
        return w, h


class TFilter:
    def __init__(self, D, U=1, cant_ma = 2):
        """
        Implementación completa de:
        T(z) = z^{-(D-1)*U} - Tₘₐ²(z)
        """
        self.samp_avg = D
        self.upsample = U
        self.cant_ma = (cant_ma//2) * 2
        self.samp_avgelay = int( self.cant_ma/2*(D - 1) * U )

        t_ma = [MovingAverageFilter(D, U)]

        for ii in range(1, self.cant_ma):
            
            t_ma += [MovingAverageFilter(D, U)]
        
        self.t_ma = t_ma
        
    def process(self, x):
        # Aplicar retardo (D-1)*U
        delayed_x = np.roll(x, self.samp_avgelay)
        delayed_x[:self.samp_avgelay] = 0

        t_ma_out = np.zeros((len(x), self.cant_ma))
        
        # Procesar con Tₘₐ²
        t_ma_out[:,0] = self.t_ma[0].process(x)
        
        for ii in range(1, self.cant_ma):
            
            t_ma_out[:,ii] = self.t_ma[ii].process(t_ma_out[:,ii-1])
        
        return delayed_x - t_ma_out[:,-1]
    
    def impulse_response(self, length=100):
        impulse = np.zeros(length)
        impulse[0] = 1
        return self.process(impulse)
    
    def frequency_response(self, n_freq=1000):
        
        w, h_ma_sq = self.t_ma[0].frequency_response(n_freq)
        h_delay = np.exp(-1j * w * self.samp_avgelay)
        
        return w, h_delay - h_ma_sq**self.cant_ma


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
    plt.figure(figsize=(14, 8))
    
    # Respuestas
    plt.subplot(3, 1, 1)
    plt.stem(h_empirical, markerfmt='bo', linefmt='b-', basefmt=' ', label='Empírica')
    plt.title(f'Comparación de Respuestas al Impulso (MA_avg={D}, MA_stages = {cant_ma}, Upsample={U} )')
    
    plt.ylabel('Empírica')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.stem(h_theoretical, markerfmt='ro', linefmt='r-', basefmt=' ', label='Teórica')
    plt.ylabel('Teórica')
    plt.grid(True, alpha=0.3)
    
    # Error
    plt.subplot(3, 1, 3)
    plt.stem(error, markerfmt='go', linefmt='g-', basefmt=' ')
    plt.ylabel('Error absoluto')
    plt.xlabel('Muestras')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Métricas de error
    print(f"\nMétricas de error para D={D}, U={U}:")
    print(f"Error máximo: {np.max(error):.2e}")
    print(f"Error RMS: {np.sqrt(np.mean(error**2)):.2e}")
    print(f"Error relativo medio: {np.mean(np.abs(error)/np.abs(h_theoretical + 1e-12)):.2e}")
    

# Ejemplo de uso
if __name__ == "__main__":

    fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
    
    dd = 32
    uu = 20
    ma_st = 4
   
    # bDebug = True
    bDebug = False
    
    if bDebug:
    
        # Ejemplo de uso
                
        # t_filter = ECGcomb(samp_avg = dd, upsample = uu, cant_ma = ma_st )
        t_filter = TFilter(dd, uu, cant_ma = ma_st )
    
        freq_resp_length = 2**16
        w_emp_fft, H_emp_fft = empirical_frequency_response(t_filter,fs = fs, signal_length = freq_resp_length)
        w_theoretical, H_theoretical = t_filter.frequency_response(n_freq = freq_resp_length)
        w_theoretical = w_theoretical/np.pi*fs/2
    
        
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
        
        mat_struct = sio.loadmat('ecg.mat')
        
        ecg_one_lead = mat_struct['ecg_lead']
        ecg_one_lead = ecg_one_lead.flatten().astype(np.float64)
        cant_muestras = len(ecg_one_lead)

        # t_filter = TFilter(dd, uu, cant_ma = ma_st )
        t_filter = ECGcomb(samp_avg = dd, upsample = uu, cant_ma = ma_st, batch = int(np.ceil(cant_muestras/5)) )
        
        # demora teórica del filtro de Rick
        demora_rl = int((dd-1)/2*ma_st*uu)
        
        ECG_f_rl_fin = t_filter.process(ecg_one_lead)
        
        # Nota: se obtuvo esta performance en una PC de escritorio estandard con:
        # Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
        # RAM: 8GB
        # Manufacturer: Gigabyte Technology Co., Ltd.
        # Product Name: B85M-D3H
        # 1129116 muestras de ECG a fs = 1kHz
        # %timeit ECG_f_rl_fin = filtro_peine_DCyArmonicas( ecg_one_lead, DD = dd, UU = uu, MA_stages = ma_st )
        # 2.01 s ± 73.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        # ECG_f_rl_fin = filtro_Lyons_opt( ecg_one_lead, DD = dd, UU = uu, MA_stages = ma_st )
        
        plt.close('all')
        
        regs_interes = ( 
                
                np.array([0, 1]) *60*fs, # minutos a muestras
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
        
                
    