# import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from SymMNA import smna
import sympy as sp

from SLiCAP import *

import platform
import subprocess
import os

from pytc2.sistemas_lineales import parametrize_sos
from pytc2.general import s, print_latex, a_equal_b_latex_s


ltspice_bin = os.path.expanduser('~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe')

fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/GIC bicuad.asc'

# Obtener la carpeta (directorio)
folder_name = os.path.dirname(fileName_asc)

# Obtener el nombre del archivo con la extensión
filename_with_extension = os.path.basename(fileName_asc)

# Separar el nombre del archivo de la extensión
baseFileName, extension = os.path.splitext(filename_with_extension)

fileName_netlist = os.path.join(folder_name, baseFileName + '.net')

if not os.path.exists(fileName_netlist):
    
    if platform.system() == 'Windows':
        file = file.replace('\\','\\\\')
        subprocess.run([ini.ltspice, '-netlist', file])
    else:
        home_directory = os.path.expanduser("~")
        
        # Configurar la variable de entorno WINEPREFIX
        os.environ['WINEPREFIX'] = os.path.join(home_directory, '.wine')    
        subprocess.run(['wine', ltspice_bin, '-wine', '-netlist', fileName_asc], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


with open(fileName_netlist, 'r', encoding='latin-1') as file:
    # Read the content of the file
    example_net_list = file.read()

# # Load the net list
# example_net_list = '''R1 N002 vi {Q*R/a}
# RG1 N002 N004 {R}
# RG3 N003 vo {R}
# RG4 vo N001 {R}
# RG5 N001 vi {R}
# C1 vi N002 {C}
# CG2 N004 N003 {C}
# XU1 N003 N002 vo opamp Aol=100K GBW=10Meg
# XU2 N003 N001 N004 opamp Aol=100K GBW=10Meg
# V1 vi 0 AC 1 1
# R2 0 N002 {Q*R/(1-a)}'''

# example_net_list_orig = '''R1 2 6 2
# RG1 2 4 1
# RG3 3 5 2
# RG4 5 1 1
# RG5 1 6 1
# C1 6 2 1
# CG2 4 3 1
# O1 3 2 5
# O2 3 1 4
# V1 6 0 1
# R2 0 2 2'''


node_names, dic_comp_name_vals, df, df2, A, X, Z, dic_params = smna(example_net_list)

# node_names, report, df0, df20, A0, X0, Z0 = smna(example_net_list_orig)
# X0 = sp.Matrix(X0)
# Z0 = sp.Matrix(Z0)
# equ0 = sp.Eq(A0*X0,Z0)

# _, df0, df20, A0, X0, Z0 = smna_orig(example_net_list)

def convert_index_to_name(x): 
    if pd.isna(x):
        return x
    return node_names[int(x)]

def translate_node_names(df_in):

    df_out = df_in.copy()
    df_out['p node'] = df['p node'].apply(convert_index_to_name)
    df_out['n node'] = df['n node'].apply(convert_index_to_name)
    df_out['Vout'] = df['Vout'].apply(convert_index_to_name)
    
    print(df_out)
    
    return df_out

dfnodenames = translate_node_names(df)

parametros_opamp = ( 'aop', 'gbw', 'aol' )
posibles_entradas = ( 'v1', 'vi', 'vin' )
posibles_salidas = ( 'v2', 'vo', 'vout' )

_, v_in_idx, _ = np.intersect1d(node_names, posibles_entradas, return_indices=True)
_, v_out_idx, _ = np.intersect1d(node_names, posibles_salidas, return_indices=True)

v_in = X[v_in_idx[0]]
v_out = X[v_out_idx[0]]

# Put matricies into SymPy 
X = sp.Matrix(X)
Z = sp.Matrix(Z)

mna_sym = [ ii for ii in A.free_symbols ]
mna_sym_names = [ str(ii) for ii in A.free_symbols ]

_, opamp_idx, _ = np.intersect1d( mna_sym_names, parametros_opamp, return_indices=True)

aop = mna_sym[opamp_idx[0]]

A = A.subs(dic_comp_name_vals).limit(aop, sp.oo)

# tuninc a mano
mna_sym = [ ii for ii in A.free_symbols ]
mna_sym_names = [ str(ii) for ii in A.free_symbols ]

_, this_idx, _ = np.intersect1d( mna_sym_names, ('eps', ), return_indices=True)

eps = mna_sym[this_idx[0]]

A = A.subs(eps, 0)

equ = sp.Eq(A0*X,Z)
# equ = sp.Eq(A*X,Z)

u1 = sp.solve(equ,X)

H = u1[v_out] / u1[v_in]

#%%

H0 = parametrize_sos(H)[0]


#%%


# H_opamp_ideal = sp.limit(H, Aop, sp.oo)
# H_opamp_ideal = H_opamp_ideal.subs({a:1/2, c:1, r:1, q:5})

# # turn the free symbols into SymPy variables
# # sp.var(str(equ.free_symbols).replace('{','').replace('}',''))
# sp.var(str(equ.free_symbols))

# # Symbolic solution
# symbolic_solution = sp.solve(equ,X)

# # The symbolic solution for the 6ltage at node 2 is:
# # symbolic_solution[v_vo]


# # Built a python dictionary of element values
# element_value_keys = []
# element_value_values = []

# for i in range(len(df)):
#     if df.iloc[i]['element'][0] == 'F' or df.iloc[i]['element'][0] == 'E' or df.iloc[i]['element'][0] == 'G' or df.iloc[i]['element'][0] == 'H':
#         element_value_keys.append(sp.var(df.iloc[i]['element'].lower()))
#         element_value_values.append(df.iloc[i]['value'])
#         #print('{:s}:{:f},'.format(df.iloc[i]['element'].lower(),df.iloc[i]['value']))
#     else:
#         element_value_keys.append(sp.var(df.iloc[i]['element']))
#         element_value_values.append(df.iloc[i]['value'])
#         #print('{:s}:{:.4e},'.format(df.iloc[i]['element'],df.iloc[i]['value']))

# element_values = dict(zip(element_value_keys, element_value_values))

# # Numeric solution
# # Substitue the element values into the equations and solve for unknown node 6ltages and currents. Need to set the current source, I1, to zero.
# equ1a = equ.subs(element_values)

# # Solve for 6ltages and currents in terms of Laplace variable s.
# u1 = sp.solve(equ1a,X)

# AC analysis
# Solve equations a frequency of 1.491MHz or $\omega$ equal to 9.3682292e6 radians per second, s = 9.3682292e6j.
# equ1a_1rad_per_s = equ1a.subs({s:9.3682292e6j})

# ans1 = solve(equ1a_1rad_per_s,X)

# for name, value in ans1.items():
#     print('{:5s}: mag: {:10.6f} phase: {:11.5f} deg'.format(str(name),float(abs(value)),float(arg(value)*180/np.pi)))

# AC Sweep
# Looking at node 2 stage.
# H = u1[v_3]



# num, denom = fraction(H) #returns numerator and denominator

# # convert symbolic to numpy polynomial
# a = np.array(Poly(num, s).all_coeffs(), dtype=float)
# b = np.array(Poly(denom, s).all_coeffs(), dtype=float)
# system = (a, b) # system for circuit

# x = np.logspace(6, 6.5, 1000, endpoint=True)*2*np.pi
# w, mag, phase = signal.bode(system, w=x) # returns: rad/s, mag in dB, phase in deg

# fig, ax1 = plt.subplots()
# ax1.set_ylabel('magnitude, dB')
# ax1.set_xlabel('frequency, Hz')

# plt.semilogx(w/(2*np.pi), mag,'-k')    # Bode magnitude plot

# ax1.tick_params(axis='y')
# plt.grid()

# # instantiate a second y-axes that shares the same x-axis
# ax2 = ax1.twinx()
# plt.semilogx(w/(2*np.pi), phase,':',color='b')  # Bode phase plot

# ax2.set_ylabel('phase, deg',color='b')
# ax2.tick_params(axis='y', labelcolor='b')
# #ax2.set_ylim((-5,25))

# plt.title('Bode plot')
# plt.show()

# print('peak: {:.2f} dB at {:.3f} MHz'.format(mag.max(),w[np.argmax(mag)]/(2*np.pi)/1e6,))



from spicelib import AscEditor, SimRunner  # Imports the class that manipulates the asc file

sallenkey = AscEditor("/home/mariano/Escritorio/Enlace hacia spice/GIC-Fliege highpass notch.ASC.asc")  # Reads the asc file into memory

# The following lines set the default tolerances for the components
mc.set_tolerance('R', 0.01)  # 1% tolerance, default distribution is uniform
mc.set_tolerance('C', 0.1, distribution='uniform')  # 10% tolerance, explicit uniform distribution
mc.set_tolerance('V', 0.1, distribution='normal')  # 10% tolerance, but using a normal distribution

# Some components can have a different tolerance
mc.set_tolerance('R1', 0.05)  # 5% tolerance for R1 only. This only overrides the default tolerance for R1

# Tolerances can be set for parameters as well
mc.set_parameter_deviation('Vos', 3e-4, 5e-3, 'uniform')  # The keyword 'distribution' is optional
mc.prepare_testbench(num_runs=1000)  # Prepares the testbench for 1000 simulations

# Finally the netlist is saved to a file. This file contians all the instructions to run the simulation in LTspice
mc.save_netlist('./testfiles/temp/sallenkey_mc.asc')




