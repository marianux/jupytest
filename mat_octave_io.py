#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:53:39 2018

@author: mllamedo
"""

import matplotlib.pyplot as plt
import scipy.io as sio

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']

#%matplotlib inline 
#%matplotlib qt5
 
plt.figure
plt.plot(ecg_one_lead)
plt.show()

# pausa para revisar los resultados
#input('Fin! Cualquier tecla para salir.')
