#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:45:55 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve

def B1(x):
    return np.where((0 <= x) & (x < 1), 1, 0)

def B2(x):
    y = np.zeros_like(x)
    mask1 = (0 <= x) & (x < 1)
    mask2 = (1 <= x) & (x < 2)
    y[mask1] = x[mask1]
    y[mask2] = 2 - x[mask2]
    return y

def B3(x):
    y = np.zeros_like(x)
    mask1 = (0 <= x) & (x < 1)
    mask2 = (1 <= x) & (x < 2)
    mask3 = (2 <= x) & (x <= 3)
    y[mask1] = 0.5 * x[mask1]**2
    y[mask2] = 0.75 - (x[mask2] - 1.5)**2
    y[mask3] = 0.5 * (3 - x[mask3])**2
    return y


# Dominio para B3, también será para B1 y B2
N = 1000
x = np.linspace(0, 3, N)

# Calcular B2 = B1 * B1 (convolución)
B2_num = convolve(B1(x).flatten(), B1(x).flatten()) 
# Calcular B3 = B1 * B2 (convolución)
B3_num = convolve(B1(x).flatten(), B2_num.flatten())

# Gráficas
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, B1(x), 'b-', label='$B_1(x)$')
plt.title("B-spline de orden 1")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, B2_num[:N], 'g-', label='$B_2(x)$')
plt.title("B-spline de orden 2")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, B3_num[:N], 'r-', label='$B_3(x)$ (numérica)')
plt.title("B-spline de orden 3 (convolución)")
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
def wavelet_psi(x, q_k):
    psi = np.zeros_like(x)
    for k in range(4):
        psi += q_k[k] * B3(2*x - k)
    return psi


# Puntos clave para evaluar la ecuación de refinamiento
x_samples = np.array([0.5, 1.0, 1.5, 2.0])

# Matriz del sistema: B3(2x_i - k) para k = 0, 1, 2, 3
A = np.zeros((4, 4))
for i, x in enumerate(x_samples):
    for k in range(4):
        A[i, k] = B3(2 * x - k)  # Usamos la B3 analítica

# Vector b: B3(x_i)
b = B3(x_samples)

# Resolver el sistema A * p = b
p_k = np.linalg.solve(A, b)

# Definir la wavelet madre usando los p_k calculados
# q_k = (-1)**np.arange(4) * p_k[::-1] / np.sqrt(2)  # Fórmula para q_k ortogonal a ϕ(x)
q_k = np.array([1.,1., -1., -1.]) * p_k[::-1] / np.sqrt(2)  # Fórmula para q_k derivada de ϕ(x)


print("Coeficientes p_k calculados:", p_k)
print("Coeficientes q_k calculados:", q_k)

x = np.linspace(0, 3, N)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, B3_num[:N], 'b-', label='$\phi(x) = B_3(x)$')
plt.title("Función de escala $\phi(x)$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, wavelet_psi(x, q_k), 'r-', label='$\psi(x)$')
plt.title("Wavelet madre $\psi(x)$")
plt.grid(True)

plt.tight_layout()
plt.show()


#%%

from scipy.signal import freqz
from scipy.fft import fft, fftfreq


def cwt(signal, scales, wavelet_func, fs=2.0):
    """
    Compute Continuous Wavelet Transform (CWT).
    
    Parameters:
        signal (array): Input signal (length N).
        scales (array): Wavelet scales (e.g., np.arange(1, 50)).
        wavelet_func (function): Wavelet function (e.g., B3-based wavelet).
        fs (float): Sampling frequency (default=1.0).
        q_k (array): Coefficients for the wavelet (if needed).
    Returns:
        cwt_matrix (array): CWT matrix (len(scales) x N).
    """
    n = len(signal)
    cwt_matrix = np.zeros((len(scales), n))
    
    # Frecuencia de Nyquist
    nyquist = fs / 2.0
    
    for i, scale in enumerate(scales):
        # Convert scale to approximate frequency
        # freq = nyquist / scale  # Relación aproximada (depende de la wavelet)
        
        # Duración del soporte: cubrir al menos 1 ciclo completo
        # (ajustar el factor 10 según la wavelet)
        support_length = int(10 * scale * fs / nyquist)  # Normalizado por fs
        
        # Dominio temporal de la wavelet
        t_wavelet = np.linspace(0, scale * 3, support_length)
        
        # Normalizar por escala y ajustar amplitud
        wavelet = wavelet_func(t_wavelet / scale, q_k) / np.sqrt(scale)
        
        # Convolución (usando 'same' para mantener el tamaño)
        cwt_matrix[i, :] = convolve(signal, wavelet, mode='same')
    
    return cwt_matrix

# def cwt(signal, scales, wavelet_func):
#     n = len(signal)
#     cwt_matrix = np.zeros((len(scales), n))
    
#     for i, scale in enumerate(scales):
#         # Crear la wavelet escalada
#         t_wavelet = np.linspace(0, scale * 3, scale)  # Soporte adaptativo
#         wavelet = wavelet_func(t_wavelet / scale, q_k) / np.sqrt(scale)
        
#         # Convolución discreta
#         cwt_matrix[i, :] = np.convolve(signal, wavelet, mode='same')
    
#     return cwt_matrix

def generar_senal(f1=50, f2=400, center1=0.25, center2=0.75,
                       width1=0.1, width2=0.1, fs=1000., N=1000):
    
    t = np.linspace(0, (N-1)/fs, N, endpoint=False)
    
    # Señal
    def gaussian_window(center, width, t):
        return np.exp(-((t - center) ** 2) / (2 * width ** 2))
    
    sig1 = np.sin(2 * np.pi * f1 * t) * gaussian_window(center1, width1, t)
    sig2 = np.sin(2 * np.pi * f2 * t) * gaussian_window(center2, width2, t)
    signal = sig1 + sig2
    
    return t, signal


# Definir una señal de ejemplo (ej: una sinusoide con ruido)
fs = 1000
N = 1000
t, signal = generar_senal(f1=10., f2=100., center1=0.25, center2=0.75, width1=0.05, width2=0.05, N = N, fs = fs)

# t = np.arange(0, N/fs, 1/fs)
# signal[N//2] = 1.

# Escalas y cálculo
# scales = np.arange(1, 50)
scales = np.logspace(0, np.log10(2), num=50)

cwt_result = cwt(signal, scales, wavelet_psi)


# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Señal
ax1.plot(t, signal)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])

pcm = ax2.imshow(np.abs(cwt_result), extent=[0, 1, 1, 50], cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()


#%%

import pywt

t, signal = generar_senal(f1=10., f2=100., center1=0.25, center2=0.75, width1=0.05, width2=0.05, N = N, fs = fs)

# Escalas y CWT
scales = np.logspace(np.log10(2), np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

# wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
wavelet = pywt.ContinuousWavelet('mexh')

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
Δt = 1.0 / fs
frequencies = f_c / (scales * Δt)

coefficients, frec = pywt.cwt(signal, scales, wavelet, sampling_period=Δt)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Señal
ax1.plot(t, signal)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])

pcm = ax2.imshow(np.abs(coefficients),
           extent=[t[0], t[-1], frec[-1], frec[0]],  # nota el orden invertido para eje Y
           cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()
