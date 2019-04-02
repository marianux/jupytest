#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script ejemplo de uso de las herramientas de simulación basadas en Python/SciPy.

Se analiza un rotador de fase de primer orden, basado en el ejemplo de 
Agustín Alba compartido por el foro.

@author: mariano
"""
# módulo de NumPy
import numpy as np
# módulo de SciPy
from scipy import signal as sig

# módulo de MatplotLib. Visualización tipo Matlab.
from matplotlib import pyplot as plt

# un módulo adaptado a mis necesidades
from splane import bodePlot, pzmap
        
# Definimos los valores de los componentes
R1 = 1
R2 = 1
R3 = 1
C4 = 1

# Constante de ganancia del inversor'
K_inv = R2 / R1

# Cargamos la funcion transferencia

num = np.array([ 1., -K_inv / R3 * C4])
den = np.array([ 1., 1. / R3 * C4])

H = sig.TransferFunction( num, den )

# Graficamos el diagrama de polos y ceros
# Graficamos la respuesta en frecuencia para el modulo y la fase.

_, axes_hdl = bodePlot(H)

# para que se vea como uno intuye el módulo. Probar comentar las siguientes 2 líneas
plt.sca(axes_hdl[0])
plt.ylim([-1,1])

plt.gca

pzmap(H)

# Pregunta: ¿que valores deberia tener R3 y C4 para tener una rotacion de fase
# de 120° a 1000Hz la salida respecto a la entrada?

