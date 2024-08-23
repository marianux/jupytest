# import os
import sympy as sp
from pytc2.sistemas_lineales import parametrize_sos
from pytc2.general import s
from pytc2.cuadripolos import smna

# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/GIC allpass norm.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/GIC_bicuad.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/ACKMOSS bicuad.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/trafo_real_mna.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/trafo_real_mna.asc'
# fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/tee_puen_2ord_delay_eq.asc'
fileName_asc = '/home/mariano/Escritorio/Enlace hacia spice/tee_puen_2ord_delay_eq2.asc'

# symbolic MNA
equ_smna, extra_results = smna(fileName_asc, 
                               bAplicarValoresComponentes = True, 
                               bAplicarParametros = True)

# tuning a mano
A0 = extra_results['A']

if extra_results['eps'] != 0:
    A0 = A0.subs(extra_results['eps'], 0)

if extra_results['aop'] != 0:
    A0 = A0.limit(extra_results['aop'], sp.oo)

equ_smna = sp.Eq(A0*extra_results['X'], extra_results['Z'])

u1 = sp.solve(equ_smna, extra_results['X'])

H = u1[extra_results['v_out']] / u1[extra_results['v_in']]

#%%

H0 = sp.collect(sp.simplify(sp.expand(H)),s)

H0 = parametrize_sos(H)[0]

display(H0)

