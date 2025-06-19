#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:22:44 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class MovingAverageFilter:
    def __init__(self, D, U=1):
        """
        Implementación recursiva de Tₘₐ(z) = (1 - z^{-D*U})/(D*(1 - z^{-U}))
        
        Parámetros:
        D : int - Retardo base
        U : int - Factor de sobremuestreo (default=1)
        """
        self.D = D
        self.U = U
        self.effective_D = D * U
        
        # Buffers para implementación recursiva
        self.x_buffer = np.zeros(self.effective_D+1)
        self.y_buffer = np.zeros(U+1)
        self.y_prev = 0.0
        
    def process(self, x):
        
        y = np.zeros_like(x, dtype=float)
        self.x_buffer = np.zeros(self.effective_D+1)
        self.y_buffer = np.zeros(U+1)
        self.y_prev = 0.0
        
        for n in range(len(x)):
            # Actualizar buffer de entrada
            self.y_buffer = np.roll(self.y_buffer, 1)
            self.x_buffer = np.roll(self.x_buffer, 1)
            self.x_buffer[0] = x[n]

            # Calcular salida (forma recursiva)
            y[n] = (self.x_buffer[0] - self.x_buffer[self.effective_D]) + self.y_buffer[self.U]

            self.y_buffer[0] = y[n]
        
        return y / self.D

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
        b = b / self.D
        
        a = np.zeros(self.U + 1)
        a[0] = 1
        a[self.U] = -1
        
        w, h = signal.freqz(b, a, worN=n_freq)
        return w, h


class TFilter:
    def __init__(self, D, U=1):
        """
        Implementación completa de:
        T(z) = z^{-(D-1)*U} - Tₘₐ²(z)
        """
        self.D = D
        self.U = U
        self.delay = (D - 1) * U
        
        self.t_ma1 = MovingAverageFilter(D, U)
        self.t_ma2 = MovingAverageFilter(D, U)
        
    def process(self, x):
        # Aplicar retardo (D-1)*U
        delayed_x = np.roll(x, self.delay)
        delayed_x[:self.delay] = 0
        
        # Procesar con Tₘₐ²
        ma_1 = self.t_ma1.process(x)
        
        ma_squared = self.t_ma2.process(ma_1)
        
        return delayed_x - ma_squared
    
    def impulse_response(self, length=100):
        impulse = np.zeros(length)
        impulse[0] = 1
        return self.process(impulse)
    
    def frequency_response(self, n_freq=1000):
        w, h_ma_sq = self.t_ma1.frequency_response(n_freq)
        h_delay = np.exp(-1j * w * self.delay)
        return w, h_delay - h_ma_sq**2


def plot_responses(D, U=1, fs=2):
    # Crear filtros
    t_ma = MovingAverageFilter(D, U)
    t_final = TFilter(D, U)
    
    # Respuestas al impulso
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    axs[0].plot(t_ma.impulse_response(2*D*U), 'o')
    axs[0].set_title(f'Tₘₐ(z) - D={D}, U={U}')
    
    axs[1].plot(t_final.impulse_response(2*D*U), 'o')
    axs[1].set_title(f'T(z) - D={D}, U={U}')
    
    plt.tight_layout()
    plt.show()
    
    # Respuestas en frecuencia
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    for filt, label in [(t_ma, 'Tₘₐ'), (t_final, 'T(z)')]:
        w, h = filt.frequency_response()
        w = w/np.pi*fs
        axs[0].plot(w , 20*np.log10(np.abs(h)), label=label)
        axs[1].plot(w, np.angle(h), label=label)
    
    axs[0].set_ylabel('Magnitud (dB)')
    axs[0].set_ylim(-60, 5)
    axs[1].set_ylabel('Fase (rad)')
    axs[1].set_xlabel('Frecuencia normalizada')
    axs[0].legend()
    axs[1].legend()
    plt.suptitle(f'Respuestas en Frecuencia - D={D}, U={U}')
    plt.show()




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
    discard = min(t_filter.D * t_filter.U, signal_length // 10)
    x = x[discard:]
    y = y[discard:]
    
    # Calcular densidad espectral cruzada y autoespectro
    f, Pxy = signal.csd(x, y, nperseg=len(x)//5, fs = fs)
    f, Pxx = signal.welch(x, nperseg=len(x)//5, fs = fs)
    
    # Respuesta en frecuencia estimada
    H_empirical = Pxy / Pxx
    
    return f, H_empirical

   
def theoretical_impulse_response(D, U=1, length=100):
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
    delay_pos = (D - 1) * U
    if delay_pos < length:
        h[delay_pos] = 1
    
    # Componente de Tₘₐ²(z)
    # Calculamos la respuesta al impulso de Tₘₐ² como la convolución de dos promedios móviles
    ma_response = np.zeros(length)
    ma_response[:D*U:U] = 1/(D)  # Respuesta de un Tₘₐ
    
    # Convolucionamos para obtener Tₘₐ²
    ma_squared_response = np.convolve(ma_response, ma_response)[:length]
    
    # Combinamos según T(z) = retardo - Tₘₐ²
    h_theoretical = h - ma_squared_response
    
    return h_theoretical

def compare_impulse_responses(filter_obj, D, U, length=60):
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
    h_theoretical = theoretical_impulse_response(D, U, length)
    
    # Calculamos el error
    error = np.abs(h_empirical - h_theoretical)
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    
    # Respuestas
    plt.subplot(3, 1, 1)
    plt.stem(h_empirical, markerfmt='bo', linefmt='b-', basefmt=' ', label='Empírica')
    plt.title(f'Comparación de Respuestas al Impulso (D={D}, U={U})')
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
    D=64
    U=20
    
    # Ejemplo de uso
    # plot_responses(D=16, U=1, fs = 1000)
   
    t_filter = TFilter(D, U)

    w_emp_fft, H_emp_fft = empirical_frequency_response(t_filter,fs = fs, signal_length = 2**16)
    w_theoretical, H_theoretical = t_filter.frequency_response()
    w_theoretical = w_theoretical/np.pi*fs/2

    
    # Gráficos comparativos
    plt.figure(figsize=(14, 8))

    plt.plot(w_theoretical, 20*np.log10(np.abs(H_theoretical)), 'b-', label='Teórica')
    plt.plot(w_emp_fft, 20*np.log10(np.abs(H_emp_fft)), 'r--', label='Empírica (FFT)')
    plt.title(f'Respuesta en Frecuencia - Magnitud (D={D}, U={U})')
    plt.ylabel('Magnitud (dB)')
    plt.legend()
    plt.grid(True)


    compare_impulse_responses(t_filter, D, U, length=4*U*D)
