#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:03:25 2023

@author: mariano
"""
# Importamos las funciones de PyTC2

# Módulos externos
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import TransferFunction
from IPython.display import display, Math, Markdown
fig_sz_x = 8
fig_sz_y = 6
fig_dpi = 150 # dpi

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi


from pytc2.sistemas_lineales import pzmap, GroupDelay, bodePlot, pretty_print_bicuad_omegayq
from pytc2.general import print_subtitle

################
## simulación ##
################

def my_experiment( QQset ):
    # Cantidad de iteraciones o experimentos
    NN = 100
    # Tolerancia de los componentes
    tol_R = 5
    tol_C = 10

    # Q y \omega_0 proyectados
    QQ = QQset
    W0 = 1

    # Valores de los componentes 
    CC = 1
    RR = 1
    RA = RR
    # si R1 = R2
    #RB = (2-1/QQ)*RA
    R1 = RR/2
    R2 = 1/(R1*W0**2*CC**2)
    KK = 2-(np.sqrt(1/R1/R2)-QQ/R1)/(QQ/R2)
    RB = RA * KK - RA
    
    # Valores de los componentes para cada iteración:
    # Cada valor es muestreado independientemente de una distribución uniforme,
    # limitada por la tolerancia impuesta.
    
    # Recordar que C es el valor que asumirá tanto C1 como C2, pero al momento de armar
    # el circuito, ambos valores son independientes.
    all_C1 = np.random.uniform(CC * (100-tol_C/2)/100 , CC * (100+tol_C/2)/100, size=NN )
    all_C2 = np.random.uniform(CC * (100-tol_C/2)/100 , CC * (100+tol_C/2)/100, size=NN )
    
    # Recordar que R es el valor que asumirá tanto R1, R2 y Ra, pero al momento de armar
    # el circuito, los valores son independientes.
    all_R1 = np.random.uniform(R1 * (100-tol_R/2)/100 , R1 * (100+tol_R/2)/100, size=NN )
    all_R2 = np.random.uniform(R2 * (100-tol_R/2)/100 , R2 * (100+tol_R/2)/100, size=NN )
    all_RA = np.random.uniform(RA * (100-tol_R/2)/100 , RA * (100+tol_R/2)/100, size=NN )
    all_RB = np.random.uniform(RB * (100-tol_R/2)/100 , RB * (100+tol_R/2)/100, size=NN )

    plt.close('all')

    fig_hdl = plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    axes_hdl = fig_hdl.subplots(2, 1, sharex='col')
    fig_id = fig_hdl.number

    # analizaremos cada iteración resultante
    #for (this_C, this_R, this_RB) in zip( all_C, all_R, all_RB):
    for ii in range(NN):

        this_KK = 1 + all_RB[ii]/all_RA[ii]
        
        this_QQ = np.sqrt(1/(all_R1[ii]*all_R2[ii]))/( 1/all_R1[ii] + (2-this_KK)/all_R2[ii] )
        this_w0 = 1/np.sqrt(all_R1[ii]*all_R2[ii]*all_C1[ii]*all_C2[ii])

        num = [this_KK * (this_w0**2)]
        den = [1, this_w0/this_QQ, this_w0**2]

        my_tf = TransferFunction( num, den )

        _, axes_hdl = bodePlot(my_tf, fig_id)


    # visualizamos la última realización a modo de ejemplo
    print_subtitle('Transferencia sampleada al azar para Q={:d}'.format(QQset))

    pretty_print_bicuad_omegayq(num,den)

    # finalmente ploteamos también la transferencia con los valores esperados
    # sin incertidumbre alguna sobre sus valores.
    KK = 1 + RB/RA

    # si R1 = R2
    #QQ = 1/(3-KK)
    # si R1/2 = R2
    #QQ = np.sqrt(2)/(4-KK)
    QQ = np.sqrt(1/R1/R2)/(1/R1+1/R2*(2-KK))
    WW0 = 1/np.sqrt(R1*R2*CC*CC)

    num = [KK * (WW0**2)]
    den = [1, WW0/QQ, WW0**2]

    # visualizamos la transferencia esperada o media
    print_subtitle('Transferencia deseada para Q={:d}'.format(QQset))

    pretty_print_bicuad_omegayq(num,den)

    my_tf = TransferFunction( num, den )

    w, mag, phase = my_tf.bode(n=300)

    (mag_ax_hdl, phase_ax_hdl) = axes_hdl

    plt.sca(mag_ax_hdl)
    plt.semilogx(w, mag, '-r', label = 'esperado', linewidth=3 )    # Bode magnitude plot

    plt.title('Magnitude Q={:d}'.format(QQset))
    plt.legend()
    plt.ylim([-20, 50])

    plt.sca(phase_ax_hdl)
    plt.semilogx(w, phase*np.pi/180, '-r', linewidth=3)    # Bode phase plot
    plt.xlim([.1, 10])
    plt.show()
    
    return( all_C1, all_C2, all_R1, all_R2, all_RA, all_RB )

# enlace para presentar los resultados.
display(Markdown(r'<a id=''2.b''></a>'))

all_Qset = [8, 8, 8]

all_C1, all_C2, all_R1, all_R2, all_RA, all_RB = my_experiment( QQset = all_Qset[0] )

results = np.vstack([all_C1, all_C2, all_R1, all_R2, all_RA, all_RB]).transpose()
list_results = [results]

for ii in range(1, len(all_Qset)):
    all_C1, all_C2, all_R1, all_R2, all_RA, all_RB = my_experiment( QQset = all_Qset[ii] )
    results = np.vstack([all_C1, all_C2, all_R1, all_R2, all_RA, all_RB]).transpose()
    list_results += [results]
    
results = np.stack( list_results, axis=2 )

