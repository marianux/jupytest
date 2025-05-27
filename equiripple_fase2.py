#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:03:52 2025

@author: mariano
"""


import numpy as np
from scipy.signal import besselap, zpk2tf, freqs
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import sos2tf_analog

# Parámetros de diseño
N       = 6             # orden total
err_perc= 5.0           # error porcentual máximo
band    = (0.0, 10.0)   # banda de diseño en rad/s
max_iter= 200

M2 = N // 2

# 1) malla de frecuencias
ww = np.linspace(band[0], band[1], 1000)

def allpass_sos(params):
    """Construye un array de SOS todo-paso a partir de params."""
    sos = []
    for k in range(M2):
        Q, w0 = params[2*k], params[2*k+1]
        # sección 2º orden: (s^2 - s w0/Q + w0^2)/(s^2 + s w0/Q + w0^2)
        sos.append([1, -w0/Q, w0**2, 1,  w0/Q, w0**2])
    return np.array(sos)

def cost(params):
    sos = allpass_sos(params)
    tf_analog = sos2tf_analog(sos)
    w, h = freqs(tf_analog.num, tf_analog.den, worN=ww)
    phase = np.unwrap(np.angle(h))
    tau = -np.gradient(phase, w)
    # error porcentual: max |tau - tau_tgt| / |tau_tgt|
    # err = np.max(np.abs((tau - tau_tgt) / tau_tgt)) * 100.0
    # # penalizamos si supera err_perc
    # if err > err_perc:
    #     return 1e6 + err**2
    # # sino minimizamos MSE
    return np.mean((tau - tau_tgt)**2)

def design_allpass_equiripple(N, err_perc, band, ):
    """
    Diseña una red todo-paso de orden N (en cascada de SOS) cuya demora de grupo siga
    un retardo Bessel normalizado con oscilación -cos(n·ω), respetando un error porcentual máximo.

    Parámetros:
    -----------
    N        : orden total del sistema (entero)
    err_perc : error porcentual máximo admisible en retardo de grupo (%)
    band     : tupla (ω_min, ω_max)
    max_iter : iteraciones máximas del optimizador

    Retorna:
    --------
    res      : resultado de la optimización (scipy OptimizeResult)
    w        : vector de frecuencias (1×1000)
    tau_opt  : retardo logrado (1×1000)
    tau_tgt  : retardo objetivo (1×1000)
    sos_opt  : arreglo de SOS optimizados (N/2 secciones × 6)
    """


    return res, w, tau_opt, tau_tgt, sos_opt



# res, w, tau_opt, tau_tgt, sos_opt = design_allpass_equiripple(N, err_perc, band)

# 2) retardo de un Bessel analógico de orden N
#    diseñamos normalizado para ω_c = 1 rad/s
z,p,k = besselap(N, norm='delay')

num, den = zpk2tf(z,p,k)

# calculamos retardo y normalizamos para τ(ω=1) = 1
w, h_b = freqs(num, den, worN=ww)
phase_b = np.unwrap(np.angle(h_b))
tau_b = -np.gradient(phase_b, w)
# tau_b /= np.interp(1.0, w, tau_b)  # normalizo a 1 en omega=1

# 3) retardo objetivo: Bessel normalizado menos cos(n·ω)
n_osc = N//2  # predicción educada: tantas oscilaciones como orden
tau_tgt = tau_b - .05 * tau_b *np.cos(n_osc * w)

# 4) parametrización de las SOS todo-paso: cada sección de 2º orden aporta (a, ω0)
#    variables = [a1, w01, a2, w02, ..., aM, w0M], M = N//2 si N par, o (N-1)//2 + 1 sección 1º orden

# inicialización Q, w0
x0 = np.tile([1.0, 1.0], M2) 
bounds = [(.1, 50.), (.1, 20.0)]*M2

# 5) optimización
res = minimize(cost, x0, bounds=bounds, method='L-BFGS-B',
               options={'maxiter': max_iter})

# Evaluamos con solución óptima
sos_opt = allpass_sos(res.x)
tf_analog = sos2tf_analog(sos_opt)
w, h_opt = freqs(tf_analog.num, tf_analog.den, worN=w)
phase_opt = np.unwrap(np.angle(h_opt))
tau_opt = -np.gradient(phase_opt, w)


print("Éxito optimización:", res.success)
print("Error porcentual alcanzado:", np.max(np.abs((tau_opt - tau_tgt)/tau_tgt))*100, "%")
print("Secciones SOS optimizadas (each row = [b0 b1 b2 a0 a1 a2]):")
print(sos_opt)

# Gráfico
plt.figure(figsize=(8,4))
plt.plot(w, tau_b, 'k:', label='τ Bessel')
plt.plot(w, tau_tgt, 'r--', label='τ objetivo (Bessel–cos)')
plt.plot(w, tau_opt, 'b', label='τ logrado')
plt.xlabel('ω (rad/s)')
plt.ylabel('Retardo de grupo')
plt.title(f'All-pass equiripple: orden {N}, error ≤ {err_perc}%')
plt.legend()
plt.grid(True)
plt.show()



