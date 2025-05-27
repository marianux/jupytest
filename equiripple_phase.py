#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:52:36 2025

@author: mariano
"""

import numpy as np
from scipy.signal import besselap, zpk2tf, freqs
from pytc2.sistemas_lineales import sos2tf_analog

from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Número de secciones todo-paso
N = 6

# Frecuencias de análisis
w = np.linspace(0, 5, 500)  # rad/s

z,p,k = besselap(N, norm='delay')

num, den = zpk2tf(z,p,k)

# calculamos retardo y normalizamos para τ(ω=1) = 1
w, h_b = freqs(num, den, worN=w)
phase_b = np.unwrap(np.angle(h_b))
tau_b = -np.gradient(phase_b, w)
# tau_b /= np.interp(1.0, w, tau_b)  # normalizo a 1 en omega=1

# 3) retardo objetivo: Bessel normalizado menos cos(n·ω)
n_osc = N//2  # predicción educada: tantas oscilaciones como orden
tau_target = tau_b - .05 * tau_b *np.cos(n_osc * w)
# tau_target = tau_b
# tau_target = np.ones_like(w) * 1.0


# Modelo de transferencia en cascada
def allpass_group_delay(params, w):
    # params = [a1, w1, a2, w2, ..., aN, wN]
    H_total = np.ones_like(w, dtype=complex)
    for i in range(0, len(params), 2):
        a = params[i]
        w0 = params[i+1]
        s = 1j * w
        num = s**2 - 2*a*s + w0**2
        den = s**2 + 2*a*s + w0**2
        H_total *= num / den
    # Fase
    phase = np.unwrap(np.angle(H_total))
    # Derivada numérica de la fase = retardo de grupo
    tau = -np.gradient(phase, w)
    return tau

# Función de costo: error cuadrático medio
def cost(params):
    tau = allpass_group_delay(params, w)
    return np.mean((tau - tau_target)**2)
    # return np.mean(np.abs(tau - tau_target))
    # return np.max(tau_target*np.abs(tau - tau_target))

# Condiciones iniciales (a_k > 0 para estabilidad, w_k > 0)
x0 = np.tile([0.5, 2.0], N)

# Restricciones: a_k > 0, w_k > 0
bounds = [(0.01, 20.0), (0.1, 30.0)] * N

# Optimización
res = minimize(cost, x0, bounds=bounds, method='L-BFGS-B')
print("Éxito:", res.success)

# Evaluar con los parámetros optimizados
tau_opt = allpass_group_delay(res.x, w)

# Gráfico
plt.plot(w, tau_target, 'k--', label='Objetivo')
plt.plot(w, tau_opt, 'b', label='Optimizado')
plt.xlabel('Frecuencia (rad/s)')
plt.ylabel('Retardo de grupo')
plt.title('Diseño de retardo de grupo equiripple (aproximado)')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Número de secciones todo-paso
N = 7

# Frecuencias de análisis
w = np.linspace(0, 10, 500)  # rad/s

# Parámetros del retardo objetivo equiripple
tau_0 = 3.5       # retardo promedio
epsilon = 0.05     # amplitud de oscilación
n = N/2             # cantidad de oscilaciones


z,p,k = besselap(N, norm='mag')
num, den = zpk2tf(z,p,k)

# calculamos retardo y normalizamos para τ(ω=1) = 1
w, h_b = freqs(num, den, worN=w)
phase_b = np.unwrap(np.angle(h_b))
tau_b = -np.gradient(phase_b, w)
# tau_b /= np.interp(1.0, w, tau_b)  # normalizo a 1 en omega=1

# 3) retardo objetivo: Bessel normalizado menos cos(n·ω)
n_osc = N/2  # predicción educada: tantas oscilaciones como orden
tau_target = tau_b - .05 * tau_b *np.cos(n_osc * w)


# Retardo de grupo deseado
# tau_target = tau_0 - epsilon * np.cos(n * w)

# Modelo de transferencia en cascada
def allpass_group_delay(params, w):
    H_total = np.ones_like(w, dtype=complex)
    for i in range(0, len(params), 2):
        a = params[i]
        w0 = params[i+1]
        s = 1j * w
        num = s**2 - 2*a*s + w0**2
        # num = w0**2
        den = s**2 + 2*a*s + w0**2
        H_total *= num / den
    # Fase
    phase = np.unwrap(np.angle(H_total))
    # Derivada numérica de la fase = retardo de grupo
    tau = -np.gradient(phase, w)
    return tau

# Función de costo: error cuadrático medio
def cost(params):
    tau = allpass_group_delay(params, w)
    return np.mean((tau - tau_target)**2)

# Condiciones iniciales: (a_k, w_k)
x0 = np.tile([0.5, 1.0], N)

# Restricciones para mantener estabilidad
bounds = [(0.01, 10.0), (0.5, 20.0)] * N

# Optimización
res = minimize(cost, x0, bounds=bounds, method='L-BFGS-B')

# Evaluar resultado
tau_opt = allpass_group_delay(res.x, w)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(w, tau_target, 'r--', label='Retardo objetivo (equiripple)')
plt.plot(w, tau_opt, 'b', label='Retardo logrado')
plt.xlabel('Frecuencia (rad/s)')
plt.ylabel('Retardo de grupo')
plt.title('Síntesis de red todo-paso con retardo equiripple')
plt.legend()
plt.grid(True)
plt.show()


#%%
