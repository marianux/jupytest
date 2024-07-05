# import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from SymMNA import smna
import sympy as sp


# Load the net list
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

example_net_list = '''R1 2 6 2
RG1 2 4 1
RG3 3 5 2
RG4 5 1 1
RG5 1 6 1
C1 6 2 1
CG2 4 3 1
O1 3 2 5
O2 3 1 4
V1 6 0 1
R2 0 2 2'''


node_names, report, df, df2, A, X, Z = smna(example_net_list)

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

# Put matricies into SymPy 
X = sp.Matrix(X)
Z = sp.Matrix(Z)

equ = sp.Eq(A*X,Z)

# turn the free symbols into SymPy variables
# sp.var(str(equ.free_symbols).replace('{','').replace('}',''))
sp.var(str(equ.free_symbols))

# Symbolic solution
symbolic_solution = sp.solve(equ,X)

# The symbolic solution for the 6ltage at node 2 is:
# symbolic_solution[v_vo]


# Built a python dictionary of element values
element_value_keys = []
element_value_values = []

for i in range(len(df)):
    if df.iloc[i]['element'][0] == 'F' or df.iloc[i]['element'][0] == 'E' or df.iloc[i]['element'][0] == 'G' or df.iloc[i]['element'][0] == 'H':
        element_value_keys.append(sp.var(df.iloc[i]['element'].lower()))
        element_value_values.append(df.iloc[i]['value'])
        #print('{:s}:{:f},'.format(df.iloc[i]['element'].lower(),df.iloc[i]['value']))
    else:
        element_value_keys.append(sp.var(df.iloc[i]['element']))
        element_value_values.append(df.iloc[i]['value'])
        #print('{:s}:{:.4e},'.format(df.iloc[i]['element'],df.iloc[i]['value']))

element_values = dict(zip(element_value_keys, element_value_values))

# Numeric solution
# Substitue the element values into the equations and solve for unknown node 6ltages and currents. Need to set the current source, I1, to zero.
equ1a = equ.subs(element_values)

# Solve for 6ltages and currents in terms of Laplace variable s.
u1 = sp.solve(equ1a,X)

# AC analysis
# Solve equations a frequency of 1.491MHz or $\omega$ equal to 9.3682292e6 radians per second, s = 9.3682292e6j.
# equ1a_1rad_per_s = equ1a.subs({s:9.3682292e6j})

# ans1 = solve(equ1a_1rad_per_s,X)

# for name, value in ans1.items():
#     print('{:5s}: mag: {:10.6f} phase: {:11.5f} deg'.format(str(name),float(abs(value)),float(arg(value)*180/np.pi)))

# AC Sweep
# Looking at node 2 stage.
# H = u1[v_3]
H = u1[v_n003]

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





