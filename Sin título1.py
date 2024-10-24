#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:35:49 2024

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from pytc2.filtros_digitales import fir_design_pm, fir_design_ls

# Parámetros comunes del derivador FIR
Ftype = 'h'  # Hilbert
order = 70  # Orden del filtro (debe ser par para tipo III)
fs = 2.0  # Frecuencia de muestreo
lgrid = 16  # Densidad del grid

# Bandas y especificaciones del derivador
band_edges = [0.05, 0.95]  # Bandas de diseño (en términos de frecuencia normalizada)
desired = [1., 1.]  # Respuesta ideal del derivador (pendiente lineal en la banda de paso)
W = [1., 1.]  # Peso en las bandas

# Diseño del filtro con PM
b_pm, _, _ = fir_design_pm(order, band_edges, desired, grid_density=lgrid, fs=fs, filter_type=Ftype)

# Diseño del filtro con LS
b_ls = fir_design_ls(order, band_edges, desired, grid_density=lgrid, fs=fs, filter_type=Ftype)

#%% MEta

b_hilbert = b_pm
# b_hilbert = b_ls

# Parámetros de la señal
fs = 100.0  # frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # vector de tiempo
fm = 5.0  # frecuencia de la señal de mensaje
fc = 20.0  # frecuencia portadora

# Señal de mensaje (secuencia arbitraria normalizada)
mensaje = np.random.uniform(-1, 1, size=int(len(t)/10))

# Interpolación de la señal de mensaje
mensaje_interp = np.interp(np.arange(0, len(mensaje), 1/fs*10), np.arange(0, len(mensaje)), mensaje)

# Modulación en amplitud
senal_am = (1 + mensaje_interp) * np.cos(2*np.pi*fc*t)

# Modulación en fase
senal_fm = np.cos(2*np.pi*fc*t + mensaje_interp)

# Aplicación del filtro FIR de Hilbert
senal_am_hilbert = np.convolve(senal_am, b_hilbert, mode='same')
senal_fm_hilbert = np.convolve(senal_fm, b_hilbert, mode='same')

# Obtención de la señal analítica
senal_am_analitica = senal_am + 1j*senal_am_hilbert
senal_fm_analitica = senal_fm + 1j*senal_fm_hilbert

# Recuperación de la señal de mensaje original
mensaje_rec_am = np.abs(senal_am_analitica) - 1
mensaje_rec_fm = np.angle(senal_fm_analitica)

# Ajuste de la demora del filtro
demora = int((len(b_hilbert)-1)/2)
mensaje_rec_am = np.roll(mensaje_rec_am, -demora)
mensaje_rec_fm = np.roll(mensaje_rec_fm, -demora)

# Gráficas
fig, axs = plt.subplots(6, 1, figsize=(10, 12))

axs[0].plot(t, mensaje_interp)
axs[0].set_title('Señal de mensaje original')

axs[1].plot(t, senal_am)
axs[1].set_title('Señal modulada en amplitud')

axs[2].plot(t, mensaje_rec_am)
axs[2].set_title('Señal de mensaje recuperada (amplitud)')

axs[3].plot(t, senal_fm)
axs[3].set_title('Señal modulada en fase')

axs[4].plot(t, mensaje_rec_fm)
axs[4].set_title('Señal de mensaje recuperada (fase)')

axs[5].plot(t, mensaje_interp - mensaje_rec_am, label='Diferencia (amplitud)')
axs[5].plot(t, mensaje_interp - mensaje_rec_fm, label='Diferencia (fase)')
axs[5].legend()
axs[5].set_title('Diferencias entre señales originales y recuperadas')

plt.tight_layout()
plt.show()

#%% Gemini


# Parámetros
fs = 1000  # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # Vector de tiempo
fc = 10  # Frecuencia de la portadora
m = np.random.uniform(-1, 1, len(t))  # Señal de mensaje

# Modulación AM y FM
s_am = (1 + m) * np.cos(2*np.pi*fc*t)
beta = 0.5  # Índice de modulación FM
s_fm = np.cos(2*np.pi*fc*t + beta*np.cumsum(m))

# Aplicando el filtro de Hilbert (suponiendo que b_hilbert está definido)
s_am_analitica = np.convolve(s_am, b_hilbert, mode='same')
s_fm_analitica = np.convolve(s_fm, b_hilbert, mode='same')

# Demodulación AM: Calculando la envolvente
m_am_demod = np.abs(s_am_analitica) - 1

# Demodulación FM: Calculando la frecuencia instantánea (aproximación)
inst_freq_fm = np.diff(np.unwrap(np.angle(s_fm_analitica))) / (2*np.pi*np.diff(t))
inst_freq_fm = np.insert(inst_freq_fm, 0, inst_freq_fm[0])
m_fm_demod = (inst_freq_fm - fc) / beta

# Comparación
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, m)
plt.title('Señal de mensaje original')

plt.subplot(3, 1, 2)
plt.plot(t, m_am_demod)
plt.title('Señal de mensaje demodulada AM')

plt.subplot(3, 1, 3)
plt.plot(t, m_fm_demod)
plt.title('Señal de mensaje demodulada FM')

plt.tight_layout()
plt.show()

#%% Claude

b_derivador = b_pm
#b_derivador = b_ls

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # Vector de tiempo de 1 segundo
fc = 50  # Frecuencia de la portadora

# Generar señal mensaje
def generar_mensaje(t):
    # Crear una señal mensaje arbitraria normalizada entre -1 y +1
    mensaje = 0.5 * np.sin(2 * np.pi * 5 * t) + \
              0.3 * np.sin(2 * np.pi * 3 * t) + \
              0.2 * np.sin(2 * np.pi * 7 * t)
    return mensaje / np.max(np.abs(mensaje))  # Normalizar

# Generar señal portadora
portadora = np.cos(2 * np.pi * fc * t)

# Generar mensaje
mensaje = generar_mensaje(t)

# Función para aplicar el filtro de Hilbert y obtener la señal analítica
def obtener_senal_analitica(senal, b_hilbert):
    # Aplicar el filtro de Hilbert
    senal_hilbert = np.convolve(senal, b_hilbert, mode='same')
    # Formar la señal analítica
    senal_analitica = senal + 1j * senal_hilbert
    return senal_analitica

# Función para visualizar resultados
def plotear_resultados(t, mensaje_original, mensaje_recuperado, titulo):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(t, mensaje_original, 'b-', label='Mensaje Original')
    plt.plot(t, mensaje_recuperado, 'r--', label='Mensaje Recuperado')
    ax.set_ylim([-2, 2])
    plt.grid(True)
    plt.legend()
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()

# 1. Modulación AM
def ejemplo_AM(mensaje, portadora, b_hilbert):
    # Generar señal AM (1 + m(t))*cos(wc*t)
    senal_AM = (1 + mensaje) * portadora
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_AM, b_hilbert)
    
    # Recuperar envolvente
    envolvente = np.abs(senal_analitica)
    
    # Normalizar y centrar la envolvente recuperada
    mensaje_recuperado = (envolvente - np.mean(envolvente)) / (np.max(envolvente) - np.min(envolvente))
    
    return mensaje_recuperado

# 2. Modulación PM
def ejemplo_PM(mensaje, portadora, b_hilbert):
    # Índice de modulación
    beta = np.pi/4
    
    # Generar señal PM cos(wc*t + β*m(t))
    senal_PM = np.cos(2*np.pi*fc*t + beta*mensaje)
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_PM, b_hilbert)
    
    # Recuperar fase instantánea
    fase_inst = np.unwrap(np.angle(senal_analitica))
    
    # Extraer mensaje (eliminar tendencia lineal de la fase)
    tendencia = np.polyfit(t, fase_inst, 1)
    fase_detrend = fase_inst - np.polyval(tendencia, t)
    
    # Normalizar mensaje recuperado
    mensaje_recuperado = fase_detrend / np.max(np.abs(fase_detrend))
    
    return mensaje_recuperado


# 3. Modulación FM (nueva)
def ejemplo_FM(mensaje, portadora, b_hilbert):
    # Índice de modulación FM
    beta = 1.
    
    # Integral del mensaje para la modulación FM
    mensaje_int = np.cumsum(mensaje) / fs
   
    # Generar señal FM cos(wc*t + β*integral(m(t)))
    senal_FM = np.cos(2*np.pi*fc*t + beta*mensaje_int)
 
    # senal_FM = np.cos(2*np.pi*(fc+beta*mensaje)*t)
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_FM, b_hilbert)
    
    # Obtener fase instantánea
    fase_inst = np.unwrap(np.angle(senal_analitica))
    
    # Derivar la fase para obtener el mensaje
    mensaje_recuperado = np.convolve(fase_inst, b_derivador, mode='same')
    
    # Normalizar
    # mensaje_recuperado = mensaje_recuperado / np.max(np.abs(mensaje_recuperado[]))
    
    parte_centrarl = mensaje_recuperado[100:-100]
    mensaje_recuperado = (mensaje_recuperado - np.mean(parte_centrarl)) / (np.max(parte_centrarl) - np.min(parte_centrarl))
    
    return mensaje_recuperado

# Ejecutar ejemplos
mensaje_recuperado_AM = ejemplo_AM(mensaje, portadora, b_hilbert)
mensaje_recuperado_PM = ejemplo_PM(mensaje, portadora, b_hilbert)
mensaje_recuperado_FM = ejemplo_FM(mensaje, portadora, b_hilbert)

# Visualizar resultados
plotear_resultados(t, mensaje, mensaje_recuperado_AM, 'Demodulación AM usando Transformada de Hilbert')
plotear_resultados(t, mensaje, mensaje_recuperado_PM, 'Demodulación PM usando Transformada de Hilbert')
plotear_resultados(t, mensaje, mensaje_recuperado_FM, 'Demodulación FM usando Transformada de Hilbert')


#%%


# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # Vector de tiempo de 1 segundo
fc = 50  # Frecuencia de la portadora

# Generar señal mensaje digital
def generar_mensaje_digital(t, rate=10):
    # Genera una secuencia binaria a la tasa especificada
    num_symbols = int(len(t) * rate / fs)
    bits = np.random.choice([0, 1], num_symbols)
    # Repetir cada bit para formar la señal
    mensaje = np.repeat(bits, fs // rate)
    # Ajustar longitud si es necesario
    if len(mensaje) < len(t):
        mensaje = np.pad(mensaje, (0, len(t) - len(mensaje)))
    else:
        mensaje = mensaje[:len(t)]
    return mensaje, bits


# 4. Modulación ASK (nueva)
def ejemplo_ASK(mensaje_digital, portadora, b_hilbert):
    # Generar señal ASK
    senal_ASK = mensaje_digital * portadora
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_ASK, b_hilbert)
    
    # Recuperar envolvente
    mensaje_recuperado = np.abs(senal_analitica)
    
    # Normalizar
    mensaje_recuperado = mensaje_recuperado / np.max(mensaje_recuperado)
    
    return mensaje_recuperado

# 5. Modulación FSK (nueva)
def ejemplo_FSK(mensaje_digital, portadora, b_hilbert):
    # Definir las dos frecuencias
    f0, f1 = fc - 10, fc + 10
    
    # Generar señal FSK
    t_matrix = np.tile(t, (2, 1))
    freq_matrix = np.array([f0, f1])[:, np.newaxis]
    bases = np.cos(2 * np.pi * freq_matrix * t_matrix)
    senal_FSK = bases[mensaje_digital.astype(int)]
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_FSK, b_hilbert)
    
    # Recuperar frecuencia instantánea
    fase_inst = np.unwrap(np.angle(senal_analitica))
    freq_inst = np.gradient(fase_inst, t) / (2 * np.pi)
    
    # Normalizar y umbralizar
    mensaje_recuperado = (freq_inst > np.mean(freq_inst)).astype(float)
    
    return mensaje_recuperado

# 6. Modulación PSK (nueva)
def ejemplo_PSK(mensaje_digital, portadora, b_hilbert):
    # Mapear bits a fases (BPSK: 0 -> 0, 1 -> π)
    beta = np.pi/2
    fases = mensaje_digital * beta - beta/2
    
    # Generar señal PSK
    senal_PSK = np.cos(2*np.pi*fc*t + fases)
    
    # Obtener señal analítica
    senal_analitica = obtener_senal_analitica(senal_PSK, b_hilbert)
    
    # Recuperar fase
    fase_inst = np.unwrap(np.angle(senal_analitica))
    
    # Extraer mensaje (eliminar tendencia lineal de la fase)
    # tendencia = np.polyfit(t, fase_inst, 1)
    tendencia = np.array([-311.31787482,    2.97861101])
    fase_detrend = fase_inst - np.polyval(tendencia, t)
    
    # Normalizar y umbralizar
    # mensaje_recuperado = (fase_detrend > 0).astype(float)
    mensaje_recuperado = fase_detrend
    
    return mensaje_recuperado


# Generar señal portadora
portadora = np.cos(2 * np.pi * fc * t)

# Generar mensajes
mensaje = generar_mensaje(t)
mensaje_digital, bits_original = generar_mensaje_digital(t)

# Ejecutar ejemplos para modulaciones digitales
mensaje_recuperado_ASK = ejemplo_ASK(mensaje_digital, portadora, b_hilbert)
# mensaje_recuperado_FSK = ejemplo_FSK(mensaje_digital, portadora, b_hilbert)
mensaje_recuperado_PSK = ejemplo_PSK(mensaje_digital, portadora, b_hilbert)

# Visualizar resultados de modulaciones digitales
plotear_resultados(t, mensaje_digital, mensaje_recuperado_ASK, 'Demodulación ASK usando Transformada de Hilbert')
# plotear_resultados(t, mensaje_digital, mensaje_recuperado_FSK, 'Demodulación FSK usando Transformada de Hilbert')
plotear_resultados(t, mensaje_digital, mensaje_recuperado_PSK, 'Demodulación PSK usando Transformada de Hilbert')
