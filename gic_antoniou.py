#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:56 2022

@author: mariano
"""

########
#%% pp #
########

import sympy as sp
# import numpy as np
# from scipy.signal import TransferFunction


# from pytc2.sintesis_dipolo import foster
from pytc2.remociones import remover_polo_dc
from pytc2.general import a_equal_b_latex_s, print_latex, s, symbfunc2tf
from pytc2.sistemas_lineales import bodePlot


I1, V1, V2, V3, V4, V5 = sp.symbols("I1, V1, V2, V3, V4, V5")
Y1, Y2, Y3, Y4, Y5, As = sp.symbols("Y1, Y2, Y3, Y4, Y5, As")
G, C, wt, w0 = sp.symbols("G, C, wt, w0", real = True, posistive = True) 

# modelo ideal negativamente realimentado
aa = sp.solve([ 
                V1*Y1 - V2*Y1 - I1, 
                -V2*Y2 + V1*(Y2+Y3) -V3*Y3,
                -V3*Y4 + V1*(Y4+Y5)
                ], 
                [V1, V2, V3])
Z1 = aa[V1]/I1

print('##############################')
print('# Z1 para el GIC de Antoniou #')
print('##############################')

# display(Math( r' Z_1^i = ' + sp.latex(Z1) ))
print_latex(a_equal_b_latex_s('Z_1^i', Z1))

######################################################
# solo modelo ideal, no sabemos qué realimentación es. 
aa = sp.solve([ 
                V1*Y1 - V2*Y1 - I1, 
                -V2*Y2 + V3*(Y2+Y3) -V4*Y3,
                -V4*Y4 + V5*(Y4+Y5),
                As*(V5-V3) - V2, 
                As*(V1-V3) - V4, 
                ], 
                [V1, V2, V3, V4, V5])

Z1 = aa[V1]/I1

# modelo ideal sin asumir realimentación negativa
Z1_opamp_ideal = sp.limit(Z1, As, sp.oo)
Z1_ideal = sp.simplify(sp.expand(Z1_opamp_ideal.subs({Y1:G, Y2:s*C, Y3:G, Y4:G, Y5:G})))

Leq_ideal = Z1_ideal/s
print_latex(a_equal_b_latex_s('L_{eq}', Leq_ideal))

Leq_ideal_val = sp.simplify(sp.expand(Leq_ideal.subs({G:1, C:1})))
print_latex(a_equal_b_latex_s('L_{eq}', Leq_ideal_val))

Z1_ideal = Z1_ideal.subs({G:1, C:1})

print('#############################################################')
print('# Z1 para el GIC de Antoniou sin considerar realim negativa #')
print('#############################################################')

# display(Math( r' Z_1^{ir} = ' + sp.latex(Z1_opamp_ideal) ))
print_latex(a_equal_b_latex_s('Z_1^{ir}', Z1_opamp_ideal))


# modelo integrador A(s)=\omega_t/s (sin asumir realimentación negativa)
Z1 = sp.simplify(sp.expand(Z1.subs(As, wt/s)))
Z1 = sp.simplify(sp.expand(Z1.subs({Y1:G, Y2:s*C, Y3:G, Y4:G, Y5:G})))
                      
num, den = sp.fraction(Z1)

num = sp.Poly(num,s)
den = sp.Poly(den,s)

k = num.LC() / den.LC()

num = num.monic()
den = den.monic()

# Implementación de un inductor mediante GIC con modelo real
Z1_opamp_real  = num/den*k

# ¿Qué tipo de Z1 será?

print('#################################################')
print('# Z1 para el GIC de Antoniou (OpAmp integrador) #')
print('#################################################')
# display(Math( r' Z_1^r = ' + sp.latex(Z1_opamp_real) ))
print_latex(a_equal_b_latex_s('Z_1^r', Z1_opamp_real))

print('¿Qué tipo de Z1 será?')

ww = sp.symbols("\omega", real = True)

# Si G/C = w0 = 1; wt = 1000 y G = 1
z1_realopamp = sp.simplify(sp.expand(Z1_opamp_real.subs({wt:100, G:1, C:1})))
# z1_realopamp = sp.simplify(sp.expand(Z1_opamp_real.subs({G:1, C:1})))
     
# analyze_sys( symbfunc2tf(z1_realopamp) )



# parte real de Z1
# re_z1_realopamp = sp.simplify(sp.expand(sp.re(z1_realopamp.subs({s:(sp.I*ww)}))))
# re_y1_realopamp = sp.simplify(sp.expand(sp.re(1/z1_realopamp.subs({s:(sp.I*ww)}))))
# re_z1_realopamp = sp.simplify(sp.expand(sp.re(Z1_opamp_real.subs({s:(sp.I*ww)}))))

# z1_realopamp_tf = symbfunc2tf(z1_realopamp)
# analyze_sys( z1_realopamp_tf )

##############
#%% análisis #
##############

def factorSOS(ratfunc):

    num, den = sp.fraction(ratfunc)
    
    num = sp.Poly(num,s)
    den = sp.Poly(den,s)

    polySOS = num.LC() / den.LC()
    
    raices = sp.roots(num, s)
    
    # Separa las raíces complejas conjugadas y las raíces reales
    raices_complejas_conjugadas_num = []
    raices_reales_num = []
    
    for raiz, multiplicidad in raices.items():
        if raiz.is_real:
            raices_reales_num.extend([raiz]*multiplicidad)
            polySOS = polySOS * (s - raiz.evalf(3))**(multiplicidad)
        else:
           # Busca si ya hay un grupo para la parte real
            grupo_existente = False
            for grupo in raices_complejas_conjugadas_num:
                # pregunto por la parte Real.
                if sp.ask(sp.Q.real((grupo + raiz))):
                    grupo_existente = True
                    break
            if not grupo_existente:
                raices_complejas_conjugadas_num.extend([raiz]*multiplicidad)
                raices_complejas_conjugadas_num.extend([sp.conjugate(raiz)]*multiplicidad)
                this_sos = sp.simplify(sp.expand((s - raiz) * (s - sp.conjugate(raiz))) )
                polySOS = polySOS * this_sos.evalf(3)**(multiplicidad)
                

    raices = sp.roots(den, s)
    
    # Separa las raíces complejas conjugadas y las raíces reales
    raices_complejas_conjugadas_den = []
    raices_reales_den = []
    
    for raiz, multiplicidad in raices.items():
        if raiz.is_real:
            raices_reales_den.extend([raiz]*multiplicidad)
            polySOS = polySOS / (s - raiz.evalf(3))**(multiplicidad)
        else:
           # Busca si ya hay un grupo para la parte real
            grupo_existente = False
            for grupo in raices_complejas_conjugadas_den:
                # pregunto por la parte Real.
                if sp.ask(sp.Q.real((grupo + raiz))):
                    grupo_existente = True
                    break
            if not grupo_existente:
                raices_complejas_conjugadas_den.extend([raiz]*multiplicidad)
                raices_complejas_conjugadas_den.extend([sp.conjugate(raiz)]*multiplicidad)
                this_sos = sp.simplify(sp.expand((s - raiz) * (s - sp.conjugate(raiz))) )
                polySOS = polySOS / this_sos.evalf(3)**(multiplicidad)


    return(polySOS, [ [raices_reales_num],[raices_reales_den] ], [[raices_complejas_conjugadas_num], [raices_complejas_conjugadas_den]])

z1_realopamp_sos, _, _ = factorSOS(z1_realopamp)
# Z1_opamp_real_sos, _, _ = factorSOS(Z1_opamp_real)

# analyze_sys( symbfunc2tf(z1_realopamp) )

##############
#%% grafs #
##############

wt_all = [10, 100, 1000]
fig_id, axes_hdl = bodePlot(symbfunc2tf(Z1_ideal), filter_description='Inductor ideal')

for wti in wt_all:

    fig_id, axes_hdl = bodePlot(symbfunc2tf(sp.simplify(sp.expand(Z1_opamp_real.subs({wt:wti, G:1, C:1})))), filter_description='$\omega_t = {:d}$'.format(wti), fig_id=1)


##############
#%% síntesis #
##############
Z1 = z1_realopamp

Y2, YLeq = remover_polo_dc(1/Z1)

Leq = 1/YLeq/s

re_Y2 = sp.simplify(sp.expand(sp.re(Y2.subs({s:(sp.I*ww)}))))

G1 = sp.minimum(re_Y2, ww, domain=sp.Reals)

R1 = 1/G1

Y4 = sp.simplify(sp.expand(Y2 - G1))

Z4 = 1/Y4

Z6, ZC1 = remover_polo_dc(Z4)

C1 = 1/ZC1/s

re_Z6 = sp.simplify(sp.expand(sp.re(Z6.subs({s:(sp.I*ww)}))))

R2 = sp.minimum(re_Z6, ww, domain=sp.Reals)

Z8 = sp.simplify(sp.expand(Z6 - R2))

Y8 = 1/Z8

Y8 =  sp.expand(Y8).as_ordered_terms()

R3 = Y8[0]
L2 = 1/Y8[1]/s

######################
#%% dibujo de la red #
######################

from schemdraw import Drawing
from pytc2.dibujar import dibujar_puerto_entrada, dibujar_funcion_exc_abajo, dibujar_elemento_serie, dibujar_elemento_derivacion, dibujar_espaciador
d = Drawing(unit=4)
d = dibujar_puerto_entrada(d)
d = dibujar_funcion_exc_abajo(d, 
                              'Z_1',  
                              z1_realopamp, 
                              hacia_salida = True)

d = dibujar_elemento_serie(d, "L", sym_label=Leq)
d = dibujar_elemento_derivacion(d, "R", sym_label=R1)
d = dibujar_elemento_serie(d, "C", sym_label=C1.evalf(3))
d = dibujar_elemento_serie(d, "R", sym_label=R2.evalf(3))
d = dibujar_elemento_derivacion(d, "R", sym_label=R3.evalf(3))
d = dibujar_espaciador(d)
d = dibujar_elemento_derivacion(d, "L", sym_label=L2.evalf(3), with_nodes=False)
display(d)
