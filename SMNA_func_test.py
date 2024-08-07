# import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from SymMNA import smna
import sympy as sp


import platform
import subprocess
import os

from pytc2.sistemas_lineales import parametrize_sos
from pytc2.general import s, print_latex, a_equal_b_latex_s


global s

ltspice_bin = os.path.expanduser('~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe')

fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/GIC allpass norm.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/GIC bicuad.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/ACKMOSS bicuad.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/trafo_real_mna.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/trafo_real_mna.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/tee_puen_2ord_delay_eq.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/tee_puen_2ord_delay_eq2.asc'



# Obtener la carpeta (directorio)
folder_name = os.path.dirname(fileName_asc)

# Obtener el nombre del archivo con la extensión
filename_with_extension = os.path.basename(fileName_asc)

# Separar el nombre del archivo de la extensión
baseFileName, extension = os.path.splitext(filename_with_extension)

fileName_netlist = os.path.join(folder_name, baseFileName + '.net')

if not os.path.exists(fileName_netlist) or ( os.path.exists(fileName_netlist) and (os.path.getmtime(fileName_asc) > os.path.getmtime(fileName_netlist))  ) :
    
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


node_names, dic_comp_name_vals, df, df2, A, X, Z, dic_params = smna(example_net_list)

# Put matricies into SymPy 
X = sp.Matrix(X)
Z = sp.Matrix(Z)

def convert_index_to_name(x): 
    if pd.isna(x):
        return x
    return node_names[int(x)]

def translate_node_names(df_in):

    df_out = df_in.copy()
    df_out['p node'] = df['p node'].apply(convert_index_to_name)
    df_out['n node'] = df['n node'].apply(convert_index_to_name)
    df_out['Vout'] = df['Vout'].apply(convert_index_to_name)
    
    # print(df_out)
    
    return df_out

dfnodenames = translate_node_names(df)

parametros_opamp = ( 'aop', 'gbw', 'aol' )
posibles_entradas = ( 'v_v1', 'v_vi', 'v_vin' )
posibles_salidas = ( 'v_v2', 'v_vo', 'v_vout' )

node_sym = [ ii for ii in X.free_symbols ]
node_sym_names = [ str(ii) for ii in X.free_symbols ]
_, v_in_idx, _ = np.intersect1d(node_sym_names, posibles_entradas, return_indices=True)
_, v_out_idx, _ = np.intersect1d(node_sym_names, posibles_salidas, return_indices=True)

v_in = node_sym[v_in_idx[0]]
v_out = node_sym[v_out_idx[0]]

mna_sym = [ ii for ii in A.free_symbols ]
mna_sym_names = [ str(ii) for ii in A.free_symbols ]

_, opamp_idx, _ = np.intersect1d( mna_sym_names, parametros_opamp, return_indices=True)

if len(opamp_idx) > 0:
    aop = mna_sym[opamp_idx[0]]
    A = A.limit(aop, sp.oo)

# aplicar parametrizaciones
A = A.subs(dic_comp_name_vals)
# A = A.subs(dic_params)

# tuning a mano
mna_sym = [ ii for ii in A.free_symbols ]
mna_sym_names = [ str(ii) for ii in A.free_symbols ]

_, this_idx, _ = np.intersect1d( mna_sym_names, ('eps', ), return_indices=True)

if len(this_idx) > 0:
    eps = mna_sym[this_idx[0]]
    A = A.subs(eps, 0)

equ = sp.Eq(A*X,Z)
# equ = sp.Eq(A*X,Z)

u1 = sp.solve(equ,X)

H = u1[v_out] / u1[v_in]

#%%

H0 = sp.collect(sp.simplify(sp.expand(H)),s)

H0 = parametrize_sos(H)[0]

