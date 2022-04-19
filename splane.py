# -*- coding: utf-8 -*-
"""

Originally based on the work of Combination of 
2011 Christopher Felton
Further modifications were added for didactic purposes
by Mariano Llamedo llamedom _at_ frba_utn_edu_ar
"""

# 2018 modified by Andres Di Donato
# 2018 modified by Mariano Llamedo Soria

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches
from collections import defaultdict
from scipy.signal import tf2zpk, TransferFunction, zpk2tf
from IPython.display import display, Math, Markdown
import sympy as sp
from schemdraw import Drawing
from schemdraw.elements import  Resistor, ResistorIEC, Capacitor, Inductor, Line, Dot, Gap, Arrow, CurrentLabelInline


def pp(z1, z2):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''

    return(z1*z2/(z1+z2))
    


'''
    Bloque de funciones para parametros imagen
'''

    

def I2T(gamma, z01, z02 = None):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    if z02 is None:
        z02 = z01

    # if np.sqrt(z02/z01)
    
    TT = np.matrix([[np.cosh(gamma)*np.sqrt(z01/z02),
                     np.sinh(gamma)*np.sqrt(z01*z02)], 
                    [np.sinh(gamma)/np.sqrt(z01*z02),
                     np.cosh(gamma)*np.sqrt(z02/z01)]])
    
    return(TT)

def I2T_s(gamma, z01, z02 = None):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    if z02 is None:
        z02 = z01
    
    TT = sp.Matrix([[sp.cosh(gamma)*sp.sqrt(z01/z02),
                     sp.sinh(gamma)*sp.sqrt(z01*z02)], 
                    [sp.sinh(gamma)/sp.sqrt(z01*z02),
                     sp.cosh(gamma)*sp.sqrt(z02/z01)]])
    
    
    return(TT)

def I2T_s(gamma, z01, z02 = None):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    if z02 is None:
        z02 = z01
    
    TT = sp.Matrix([[sp.cosh(gamma)*sp.sqrt(z01/z02),
                     sp.sinh(gamma)*sp.sqrt(z01*z02)], 
                    [sp.sinh(gamma)/sp.sqrt(z01*z02),
                     sp.cosh(gamma)*sp.sqrt(z02/z01)]])
    
    
    return(TT)


'''
    Bloque de funciones para dibujar redes de forma bonita
'''

def dibujar_puerto_entrada(d, port_name = None, voltage_lbl = None, current_lbl = None):
    
    d += Dot(open=True)
    
    if voltage_lbl is None:
        d += Gap().down().label( '' )
    else:
        d += Gap().down().label( voltage_lbl, fontsize=16)
    
    d.push()

    if not(port_name is None):
        d += Gap().left().label( '' ).length(d.unit*.35)
        d += Gap().up().label( port_name, fontsize=22)
        d.pop()
        
    d += Dot(open=True)
    d += Line().right().length(d.unit*.5)
    d += Gap().up().label( '' )
    d.push()
    
    if current_lbl is None:
        d += Line().left().length(d.unit*.5)
    else:
        d += Line().left().length(d.unit*.25)
        d += Arrow(reverse=True).left().label( current_lbl, fontsize=16).length(d.unit*.25)
    
    d.pop()

    return(d)

def dibujar_puerto_salida(d, port_name = None, voltage_lbl = None, current_lbl = None):
    
    if current_lbl is None:
        d += Line().right().length(d.unit*.5)
    else:
        d += Line().right().length(d.unit*.25)
        d += Arrow(reverse=True).right().label( current_lbl, fontsize=16).length(d.unit*.25)
    
    d += Dot(open=True)
    
    d.push()

    if voltage_lbl is None:
        d += Gap().down().label( '' )
    else:
        d += Gap().down().label( voltage_lbl, fontsize=16)


    if not(port_name is None):
        d.push()
        d += Gap().right().label( '' ).length(d.unit*.35)
        d += Gap().up().label( port_name, fontsize=22)
        d.pop()

    d += Dot(open=True)
    d += Line().left().length(d.unit*.5)

    d.pop()

    return(d)


def dibujar_espaciador( d ):

    d += Line().right().length(d.unit*.5)

    d.push()

    d += Gap().down().label( '' )

    d += Line().left().length(d.unit*.5)

    d.pop()

    return(d)


def dibujar_funcion_exc_abajo(d, func_label, sym_func, k_gap_width=0.5, hacia_salida  = False, hacia_entrada  = False ):

    half_width = d.unit*k_gap_width/2
    
    d += Line().right().length(half_width)
    d.push()
    d += Gap().down().label('')
    d.push()
    lbl = d.add(Gap().down().label( '$ ' + func_label + ' = ' + sp.latex(sym_func) + ' $', fontsize=22 ).length(0.5*half_width))
    d.pop()
    d.push()
    d += Line().up().at( (d.here.x, d.here.y - .2 * half_width) ).length(half_width).linewidth(1)
    
    if( hacia_salida ):
        d.push()
        d += Arrow().right().length(.5*half_width).linewidth(1)
        d.pop()
        
    if( hacia_entrada ):
        d += Arrow().left().length(.5*half_width).linewidth(1)
        
    d.pop()
    d.push()
    d += Line().left().length(half_width)
    d.pop()
    d += Line().right().length(half_width)
    d.pop()
    d += Line().right().length(half_width)

    return([d, lbl])

def dibujar_funcion_exc_arriba(d, func_label, sym_func, k_gap_width=0.5, hacia_salida = False, hacia_entrada = False ):

    half_width = d.unit*k_gap_width/2
    
    d += Line().right().length(half_width)
    d.push()
    lbl = d.add(Gap().up().label( '$ ' + func_label + ' = ' + sp.latex(sym_func) + ' $', fontsize=22 ).length(3* half_width))
    d.pop()
    d.push()
    d += Line().down().at( (d.here.x, d.here.y + .2 * half_width) ).length(half_width).linewidth(1)
    
    if( hacia_salida ):
        d.push()
        d += Arrow().right().length(.5*half_width).linewidth(1)
        d.pop()
        
    if( hacia_entrada ):
        d += Arrow().left().length(.5*half_width).linewidth(1)
        
    d.pop()
    d.push()
    d += Gap().down().label('')
    d.push()
    d += Line().left().length(half_width)
    d.pop()
    d += Line().right().length(half_width)
    d.pop()
    d += Line().right().length(half_width)



    return([d, lbl])

def dibujar_elemento_serie(d, elemento, sym_label=''):
    
    if isinstance(sym_label, sp.Number ):
        sym_label = to_latex(sym_label)
    elif isinstance(sym_label, np.number):
        sym_label = str_to_latex('{:3.3f}'.format(sym_label))
    elif isinstance(sym_label, str):
        sym_label = str_to_latex(sym_label)
    else:
        sym_label = '$ ?? $'

    
    d += elemento().right().label(sym_label, fontsize=16)
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_elemento_derivacion(d, elemento, sym_label=''):
    
    if isinstance(sym_label, sp.Number ):
        sym_label = to_latex(sym_label)
    elif isinstance(sym_label, np.number):
        sym_label = str_to_latex('{:3.3f}'.format(sym_label))
    elif isinstance(sym_label, str):
        sym_label = str_to_latex(sym_label)
    else:
        sym_label = '$ ?? $'
    
    d += Dot()
    d.push()
    d += elemento().down().label(sym_label, fontsize=16)
    d += Dot()
    d.pop()

    return(d)


def dibujar_tanque_serie(d, sym_ind_label='', sym_cap_label=''):
    
    if isinstance(sym_ind_label, sp.Number ):
        sym_ind_label = to_latex(sym_ind_label)
    else:
        sym_ind_label = str_to_latex(sym_ind_label)
    
    if isinstance(sym_cap_label, sp.Number ):
        sym_cap_label = to_latex(sym_cap_label)
    else:
        sym_cap_label = str_to_latex(sym_cap_label)
    
    d.push()
    d += Dot()
    d += Capacitor().right().label(sym_cap_label, fontsize=16)
    d.pop()
    d += Line().up().length(d.unit*.5)
    d += Inductor().right().label(sym_ind_label, fontsize=16)
    d += Line().down().length(d.unit*.5)
    d += Dot()
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_tanque_derivacion(d, sym_ind_label='', sym_cap_label=''):
    
    if isinstance(sym_ind_label, sp.Number ):
        sym_ind_label = to_latex(sym_ind_label)
    else:
        sym_ind_label = str_to_latex(sym_ind_label)
    
    if isinstance(sym_cap_label, sp.Number ):
        sym_cap_label = to_latex(sym_cap_label)
    else:
        sym_cap_label = str_to_latex(sym_cap_label)
    
    d.push()
    d += Dot()
    d += Capacitor().down().label(sym_cap_label, fontsize=16).length(d.unit*.5)
    d += Inductor().down().label(sym_ind_label, fontsize=16).length(d.unit*.5)
    d += Dot()
    d.pop()

    return(d)


'''
    Bloque de funciones para la síntesis gráfica de imitancias
'''

s = sp.symbols('s ', complex=True)

def remover_polo_jw( imit, omega, omega_zero = None ):
    '''
    Se removerá el residuo en sobre el eje $j.\omega$ (omega) de la imitancia 
    $I$ (imit) de forma completa, o parcial en el caso que se especifique una 
    omega_zero.
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ I_{R}=I-\frac{2.k.s}{s^{2}+\omega^{2}} $$
    
    siendo 

    $$ k=\lim\limits _{s^2\to-\omega^2}I\frac{2.k.s}{s^{2}+\omega^{2}} $$
    
    En cuanto se especifique omega_zero, la remoción parcial estará definida 
    como

    $$ I_{R}\biggr\rfloor_{s^{2}=-\omega_{z}^{2}}=0=I-\frac{2.k.s}{s^{2}+\omega^{2}}\biggr\rfloor_{s^{2}=-\omega_{z}^{2}} $$
    
    siendo 
    
    $$ 2.k^{'}=I.\frac{s^{2}+\omega^{2}}{s}\biggr\rfloor_{s^{2}=-\omega_z^{2}} $$
    

    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un polo de orden 1 en \omega.
    omega_zero : Symbolic
        Frecuencia a la que la imitancia será cero luego de la remoción.

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    if omega_zero is None:
        # remoción total
        # kk = sp.limit(imit*(s**2+omega**2)/s, s**2, -omega**2)
        kk = sp.simplify(sp.expand(imit*(s**2+omega**2)/s)).subs(s**2, -(omega**2) )
        
    else:
        # remoción parcial
        kk = sp.simplify(sp.expand(imit*(s**2+omega**2)/s)).subs(s**2, -(omega_zero**2) )
    
    # extraigo kk
    imit_r = sp.factor(sp.simplify(sp.expand(imit - kk*s/(s**2+omega**2))))

    return( [imit_r, kk] )

def remover_polo_dc( imit, omega_zero = None ):
    '''
    Se removerá el residuo en continua (s=0) de la imitancia ($I$) de forma 
    completa, o parcial en el caso que se especifique una omega_zero. 
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ I_R = I - k_0/s  $$
    
    siendo 

    $$ k_0=\lim\limits _{s\to0}I.s $$
    
    En cuanto se especifique omega_zero, la remoción parcial estará definida 
    como

    $$ I_{R}\biggr\rfloor_{s^{2}=-\omega_z^{2}}=0=I-s.k_{0}^{'}\biggr\rfloor_{s^{2}=-\omega_z^{2}} $$
    
    siendo 
    
    $$ k_{0}^{'}=I.s\biggr\rfloor_{s^{2}=-\omega_z^{2}} $$
    

    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un polo de orden 1 en 0, es decir la 
        diferencia de grados entre num y den será exactamente -1.
    omega_zero : Symbolic
        Frecuencia a la que la imitancia será cero luego de la remoción.

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    if omega_zero is None:
        # remoción total
        k_cero = sp.limit(imit*s, s, sp.oo)
        
    else:
        # remoción parcial
        k_cero = sp.simplify(sp.expand(imit*s)).subs(s**2, -(omega_zero**2) )

    
    # extraigo C3
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_cero/s)))

    return( [imit_r, k_cero] )

def remover_polo_infinito( imit, omega_zero = None ):
    '''
    Se removerá el residuo en infinito de la imitancia ($I$) de forma 
    completa, o parcial en el caso que se especifique una omega_zero. 
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ I_R = I - s.k_\infty  $$
    
    siendo 

    $$ k_{\infty}=\lim\limits _{s\to\infty}I.\nicefrac{1}{s} $$
    
    En cuanto se especifique omega_zero, la remoción parcial estará definida 
    como

    $$ I_{R}\biggr\rfloor_{s^{2}=-\omega_z^{2}}=0=I-s.k_{\infty}^{'}\biggr\rfloor_{s^{2}=-\omega_z^{2}} $$
    
    siendo 
    
    $$ k_{\infty}^{'}=I.\nicefrac{1}{s}\biggr\rfloor_{s^{2}=-\omega_z^{2}} $$
    

    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un polo de orden 1 en infinito, es decir la 
        diferencia de grados entre num y den será exactamente 1.
    omega_zero : Symbolic
        Frecuencia a la que la imitancia será cero luego de la remoción.

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    if omega_zero is None:
        # remoción total
        k_inf = sp.limit(imit/s, s, sp.oo)
        
    else:
        # remoción parcial
        k_inf = sp.simplify(sp.expand(imit/s)).subs(s**2, -(omega_zero**2) )

    
    # extraigo C3
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_inf*s)))

    return( [imit_r, k_inf] )


def tanque_z( doska, omegasq ):
    '''
    Calcula los valores de L y C que componen un tanque resonante LC 
    (tanque Z), a partir del valor del residuo ($ k $) y la omega al cuadrado 
    ($ \omega^2 $) de la expresión de impedancia dada por:
        
        $$ Z_{LC} = \frac{2.k.s}{(s^2+\omega^2)} $$

    Parameters
    ----------
    doska : Symbolic
        Dos veces el residuo.
    omegasq : Symbolic
        Cuadrado de la omega a la que el tanque resuena.

    Returns
    -------
    L : Symbolic
        Valor de la admitancia
    C : Symbolic
        Valor de la capacidad

    '''
    
    return( [doska/omegasq, 1/doska] )

def tanque_y( doska, omegasq ):
    '''
    Calcula los valores de L y C que componen un tanque resonante LC 
    (tanque Z), a partir del valor del residuo ($ k $) y la omega al cuadrado 
    ($ \omega^2 $) de la expresión de impedancia dada por:
        
        $$ Y_{LC} = \frac{2.k.s}{(s^2+\omega^2)} $$

    Parameters
    ----------
    doska : Symbolic
        Dos veces el residuo.
    omegasq : Symbolic
        Cuadrado de la omega a la que el tanque resuena.

    Returns
    -------
    L : Symbolic
        Valor de la admitancia
    C : Symbolic
        Valor de la capacidad

    '''
    
    return( [1/doska, doska/omegasq] )


def to_latex( unsimbolo ):
    '''
    Convierte un símbolo en un string formateado para visualizarse en LaTex 
    '''
    
    return('$'+ sp.latex(unsimbolo) + '$')

def str_to_latex( unstr):
    '''
    Formatea un string para visualizarse en LaTex 
    '''
    
    return('$'+ unstr + '$')


'''
    Funciones de conversión de matrices de cuadripolos lineales
'''

def Y2T_s(YY):
    
    TT = sp.Matrix([[0, 0], [0, 0]])
    
    # A = Y22/Y21
    TT[0,0] = sp.simplify(sp.expand(-YY[1,1]/YY[1,0]))
    # B = -1/Y21
    TT[0,1] = sp.simplify(sp.expand(-1/YY[1,0]))
    # C = -DY/Y21
    TT[1,0] = sp.simplify(sp.expand(-sp.Determinant(YY)/YY[1,0]))
    # D = Y11/Y21
    TT[1,1] = sp.simplify(sp.expand(-YY[1,1]/YY[1,0]))
    
    return(TT)

def Z2T_s(ZZ):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    TT = sp.Matrix([[0, 0], [0, 0]])
    
    # A = Z11/Z21
    TT[0,0] = sp.simplify(sp.expand(ZZ[0,0]/ZZ[1,0]))
    # B = DZ/Z21
    TT[0,1] = sp.simplify(sp.expand(sp.Determinant(ZZ)/ZZ[1,0]))
    # C = 1/Z21
    TT[1,0] = sp.simplify(sp.expand(1/ZZ[1,0]))
    # D = Z22/Z21
    TT[1,1] = sp.simplify(sp.expand(ZZ[1,1]/ZZ[1,0]))
    
    return(TT)

def T2Z_s(TT):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    ZZ = sp.Matrix([[0, 0], [0, 0]])
    
    # Z11 = A/C
    ZZ[0,0] = sp.simplify(sp.expand(TT[0,0]/TT[1,0]))
    # Z11 = DT/C
    ZZ[0,1] = sp.simplify(sp.expand(sp.Determinant(TT)/TT[1,0]))
    # Z21 = 1/C
    ZZ[1,0] = sp.simplify(sp.expand(1/TT[1,0]))
    # Z22 = D/C
    ZZ[1,1] = sp.simplify(sp.expand(TT[1,1]/TT[1,0]))
    
    return(ZZ)

def T2Y_s(TT):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    YY = sp.Matrix([[0, 0], [0, 0]])
    
    # Y11 = D/B
    YY[0,0] = sp.simplify(sp.expand(TT[1,1]/TT[0,1]))
    # Y12 = -DT/B
    YY[0,1] = sp.simplify(sp.expand(-sp.Determinant(TT)/TT[0,1]))
    # Y21 = -1/B
    YY[1,0] = sp.simplify(sp.expand(-1/TT[0,1]))
    # Y22 = A/B
    YY[1,1] = sp.simplify(sp.expand(TT[0,0]/TT[0,1]))
    
    return(YY)

def Y2T(YY):
    
    TT = np.zeros_like(YY)
    
    # A = Y22/Y21
    TT[0,0] = -YY[1,1]/YY[1,0]
    # B = -1/Y21
    TT[0,1] = -1/YY[1,0]
    # C = -DY/Y21
    TT[1,0] = -np.linalg.det(YY)/YY[1,0]
    # D = Y11/Y21
    TT[1,1] = -YY[1,1]/YY[1,0]
    
    return(TT)

def Z2T(ZZ):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    TT = np.zeros_like(ZZ)
    
    # A = Z11/Z21
    TT[0,0] = ZZ[0,0]/ZZ[1,0]
    # B = DZ/Z21
    TT[0,1] = np.linalg.det(ZZ)/ZZ[1,0]
    # C = 1/Z21
    TT[1,0] = 1/ZZ[1,0]
    # D = Z22/Z21
    TT[1,1] = ZZ[1,1]/ZZ[1,0]
    
    return(TT)

def T2Z(TT):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    ZZ = np.zeros_like(TT)
    
    # Z11 = A/C
    ZZ[0,0] = TT[0,0]/TT[1,0]
    # Z11 = DT/C
    ZZ[0,1] = np.linalg.det(TT)/TT[1,0]
    # Z21 = 1/C
    ZZ[1,0] = 1/TT[1,0]
    # Z22 = D/C
    ZZ[1,1] = TT[1,1]/TT[1,0]
    
    return(ZZ)

def T2Y(TT):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    YY = np.zeros_like(TT)
    
    # Y11 = D/B
    YY[0,0] = TT[1,1]/TT[0,1]
    # Y12 = -DT/B
    YY[0,1] = -np.linalg.det(TT)/TT[0,1]
    # Y21 = -1/B
    YY[1,0] = -1/TT[0,1]
    # Y22 = A/B
    YY[1,1] = TT[0,0]/TT[0,1]
    
    return(YY)


def Z2tee(ZZ):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    # Dibujo la red Tee
    
    d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads
    
    d = dibujar_puerto_entrada(d,
                                   port_name = 'In', 
                                   voltage_lbl = ('+', '$V_1$', '-'), 
                                   current_lbl = '$I_1$')
    
    Za = ZZ[0,0] - ZZ[0,1] 
    Zb = ZZ[0,1] 
    Zc = ZZ[1,1] - ZZ[0,1] 
    
    d = dibujar_elemento_serie(d, ResistorIEC, Za )
    d = dibujar_elemento_derivacion(d, ResistorIEC, Zb )
    d = dibujar_elemento_serie(d, ResistorIEC, Zc )
    
    d = dibujar_puerto_salida(d, 
                                  port_name = 'Out', 
                                  current_lbl = '$I_2$' )

    display(d)        
    
    return([Za,Zb,Zc])


def Y2Pi(YY):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    # Dibujo la red Tee
    
    d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads
    
    d = dibujar_puerto_entrada(d,
                                   port_name = 'In', 
                                   voltage_lbl = ('+', '$V_1$', '-'), 
                                   current_lbl = '$I_1$')
    
    Ya = YY[0,0] + YY[0,1]
    Yb = -YY[0,1]
    Yc = YY[1,1] + YY[0,1]
    
    if isinstance(YY[0,0], sp.Symbol):
        
        Za = sp.simplify(sp.expand(1/Ya))
        Zb = sp.simplify(sp.expand(1/Yb))
        Zc = sp.simplify(sp.expand(1/Yc))
        
    else:

        Za = 1/(YY[0,0] + YY[0,1])
        Zb = 1/(-YY[0,1])
        Zc = 1/(YY[1,1] + YY[0,1])
    
    d = dibujar_elemento_derivacion(d, ResistorIEC, Za )
    d = dibujar_elemento_serie(d, ResistorIEC, Zb )
    d = dibujar_elemento_derivacion(d, ResistorIEC, Zc )
    
    d = dibujar_puerto_salida(d, 
                                  port_name = 'Out', 
                                  current_lbl = '$I_2$' )
    
    display(d)        
    
    return([Ya, Yb, Yc])




def y2mai(YY):
    '''
    Convierte la MAD en MAI luego de levantar de referencia.

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    Ymai = YY.row_insert(YY.shape[0], sp.Matrix([-sum(YY[:,ii] ) for ii in range(YY.shape[1])]).transpose() )
    Ymai = Ymai.col_insert(Ymai.shape[1], sp.Matrix([-sum(Ymai[ii,:] ) for ii in range(Ymai.shape[0])]) )
    Ymai[-1] = sum(YY)
    
    return(Ymai)

def may2y(Ymai, nodes2del):
    '''
    Convierte la MAI en MAD luego de remover filas y columnas indicadas en nodes2del

    Parameters
    ----------
    Ymai : Symbolic Matrix
        Matriz admitancia indefinida.
    nodes2del : list or integer
        Nodos que se van a eliminar.

    Returns
    -------
    YY : Symbolic Matrix
        Matriz admitancia 

    '''
    
    YY = Ymai
    
    for ii in nodes2del:
        YY.row_del(ii)
    
    for ii in nodes2del:
        YY.col_del(ii)
    
    return(YY)


def calc_MAI_ztransf_ij_mn(Ymai, ii=2, jj=3, mm=0, nn=1, verbose=False):
    """
    Calcula la transferencia de impedancia V_ij / I_mn
    """
    
    if ii > jj:
        max_ouput_idx = ii
        min_ouput_idx = jj
    else:
        max_ouput_idx = jj
        min_ouput_idx = ii
    
    if mm > nn:
        max_input_idx = mm
        min_input_idx = nn
    else:
        max_input_idx = nn
        min_input_idx = mm
    
    # cofactor de 2do orden
    num = Ymai.minor_submatrix(max_ouput_idx, max_input_idx).minor_submatrix(min_ouput_idx, min_input_idx)
    # cualquier cofactor de primer orden
    den = Ymai.minor_submatrix(min_input_idx, min_input_idx)

    num_det = sp.simplify(num.det())
    den_det = sp.simplify(den.det())
    
    sign_correction = mm+nn+ii+jj
    Tz = sp.simplify(-1**(sign_correction) * num_det/den_det)
    
    if( verbose ):
    
        print_latex(r' [Y_{MAI}] = ' + sp.latex(Ymai) )
        
        print_latex(r' [Y^{{ {:d}{:d} }}_{{ {:d}{:d} }} ] = '.format(mm,nn,ii,jj) + sp.latex(num) )
    
        print_latex(r'[Y^{{ {:d} }}_{{ {:d} }}] = '.format(mm,mm) + sp.latex(den) )
    
        print_latex(r'\mathrm{{Tz}}^{{ {:d}{:d} }}_{{ {:d}{:d} }} = \frac{{ \underline{{Y}}^{{ {:d}{:d} }}_{{ {:d}{:d} }} }}{{ \underline{{Y}}^{{ {:d} }}_{{ {:d} }} }} = '.format(ii,jj,mm,nn,mm,nn,ii,jj,mm,mm) + r' -1^{{ {:d} }} '.format(sign_correction)  + r'\frac{{ ' + sp.latex(num_det) + r'}}{{' + sp.latex(den_det) + r'}} = ' + sp.latex(Tz))
    
    return(Tz)

def calc_MAI_vtransf_ij_mn(Ymai, ii=2, jj=3, mm=0, nn=1, verbose=False):
    """
    Calcula la transferencia de tensión V_ij / V_mn
    """
    
    if ii > jj:
        max_ouput_idx = ii
        min_ouput_idx = jj
    else:
        max_ouput_idx = jj
        min_ouput_idx = ii
    
    if mm > nn:
        max_input_idx = mm
        min_input_idx = nn
    else:
        max_input_idx = nn
        min_input_idx = mm
    
    # cofactores de 2do orden
    num = Ymai.minor_submatrix(max_ouput_idx, max_input_idx).minor_submatrix(min_ouput_idx, min_input_idx)

    den = Ymai.minor_submatrix(max_input_idx, max_input_idx).minor_submatrix(min_input_idx, min_input_idx)
    
    num_det = sp.simplify(num.det())
    den_det = sp.simplify(den.det())
    
    sign_correction = mm+nn+ii+jj
    Av = sp.simplify(-1**(sign_correction) * num_det/den_det)
    
    if( verbose ):
    
        print_latex(r' [Y_{MAI}] = ' + sp.latex(Ymai) )
        
        print_latex(r' [Y^{{ {:d}{:d} }}_{{ {:d}{:d} }} ] = '.format(mm,nn,ii,jj) + sp.latex(num) )
    
        print_latex(r'[Y^{{ {:d}{:d} }}_{{ {:d}{:d} }} ] = '.format(mm,nn,mm,nn) + sp.latex(den) )
    
        print_latex(r'T^{{ {:d}{:d} }}_{{ {:d}{:d} }} = \frac{{ \underline{{Y}}^{{ {:d}{:d} }}_{{ {:d}{:d} }} }}{{ \underline{{Y}}^{{ {:d}{:d} }}_{{ {:d}{:d} }} }} = '.format(ii,jj,mm,nn,mm,nn,ii,jj,mm,nn,mm,nn) + r' -1^{{ {:d} }} '.format(sign_correction)  + r'\frac{{ ' + sp.latex(num_det) + r'}}{{' + sp.latex(den_det) + r'}} = ' + sp.latex(Av) )
    
    return(Av)


def calc_MAI_impedance_ij(Ymai, ii=0, jj=1, verbose=False):
    
    if ii > jj:
        max_idx = ii
        min_idx = jj
    else:
        max_idx = jj
        min_idx = ii
 
    # cofactor de 2do orden
    num = Ymai.minor_submatrix(max_idx, max_idx).minor_submatrix(min_idx, min_idx)
    # cualquier cofactor de primer orden
    den = Ymai.minor_submatrix(min_idx, min_idx)
    
    ZZ = sp.simplify(num.det()/den.det())
    
    if( verbose ):

        print_latex(r' [Y_{MAI}] = ' + sp.latex(Ymai) )
        
        print_latex(r' [Y^{{ {:d}{:d} }}_{{ {:d}{:d} }} ] = '.format(ii,ii,jj,jj) + sp.latex(num) )

        print_latex(r'[Y^{{ {:d} }}_{{ {:d} }}] = '.format(ii,ii) + sp.latex(den) )

        print_latex(r'Z_{{ {:d}{:d} }} = \frac{{ \underline{{Y}}^{{ {:d}{:d} }}_{{ {:d}{:d} }} }}{{ \underline{{Y}}^{{ {:d} }}_{{ {:d} }} }} = '.format(ii,jj,ii,ii,jj,jj,ii,ii) + sp.latex(ZZ))

    return(ZZ)



'''
Otras funciones

'''

def modsq2mod_s( aa ):

    num, den = sp.fraction(aa)
    
    roots_num = sp.solve(num)
    
    real_part_roots = [ (s-roots_numm) for roots_numm in roots_num if sp.re(roots_numm) <= 0]

    poly_acc = sp.Rational('1')
    
    for roots_numm in real_part_roots:
        poly_acc *= roots_numm
    
    num = sp.simplify(sp.expand(poly_acc))

    roots_num = sp.solve(den)
    
    real_part_roots = [ (s-roots_numm) for roots_numm in roots_num if sp.re(roots_numm) <= 0]

    poly_acc = sp.Rational('1')
    
    for roots_numm in real_part_roots:
        poly_acc *= roots_numm
    
    poly_acc = sp.simplify(sp.expand(poly_acc))

    return(sp.simplify(sp.expand(num/poly_acc))) 


def modsq2mod( aa ):
    
    rr = np.roots(aa)
    bb = rr[np.real(rr) == 0]
    bb = bb[ :(bb.size//2)]
    bb = np.concatenate( [bb, rr[np.real(rr) < 0]])
    
    return np.flip(np.real(np.polynomial.polynomial.polyfromroots(bb)))

def tfcascade(tfa, tfb):

    tfc = TransferFunction( np.polymul(tfa.num, tfb.num), np.polymul(tfa.den, tfb.den) )

    return tfc

def tfadd(tfa, tfb):

    tfc = TransferFunction( np.polyadd(np.polymul(tfa.num,tfb.den),np.polymul(tfa.den,tfb.num)),
                            np.polymul(tfa.den,tfb.den) )
    return tfc


def build_poly_str(this_poly):
    
    poly_str = ''

    for ii in range( this_poly.shape[0] ):
    
        if this_poly[ii] != 0.0:
            
            if (this_poly.shape[0]-2) == ii:
                poly_str +=  '+ s ' 
            
            elif (this_poly.shape[0]-1) != ii:
                poly_str +=  '+ s^{:d} '.format(this_poly.shape[0]-ii-1) 

            if (this_poly.shape[0]-1) == ii:
                poly_str += '+ {:3.4g} '.format(this_poly[ii])
            else:
                if this_poly[ii] != 1.0:
                    poly_str +=  '\,\, {:3.4g} '.format(this_poly[ii])
                
    return poly_str[2:]

def build_omegayq_str(this_quad_poly, den = np.array([])):

    if den.shape[0] > 0:
        # numerator style bandpass s. hh . oemga/ qq
        
        omega = np.sqrt(den[2]) # from denominator
        qq = omega / den[1] # from denominator
        
        hh = this_quad_poly[1] * qq / omega
        
        poly_str = r's\,{:3.4g}\,\frac{{{:3.4g}}}{{{:3.4g}}}'.format(hh, omega, qq )
    
    else:
        # all other complete quadratic polynomial
        omega = np.sqrt(this_quad_poly[2])
        qq = omega / this_quad_poly[1]
        
        poly_str = r's^2 + s \frac{{{:3.4g}}}{{{:3.4g}}} + {:3.4g}^2'.format(omega, qq, omega)
                
    return poly_str

def print_console_alert(strAux):
    
    strAux = '# ' + strAux + ' #\n'
    strAux1 =  '#' * (len(strAux)-1) + '\n' 
    
    print( '\n\n' + strAux1 + strAux + strAux1 )
    
def print_console_subtitle(strAux):
    
    strAux = strAux + '\n'
    strAux1 =  '-' * (len(strAux)-1) + '\n' 
    
    print( '\n\n' + strAux + strAux1 )
    
def print_subtitle(strAux):
    
    display(Markdown('#### ' + strAux))

def print_latex(strAux):
    
    display(Math(strAux))


def pretty_print_lti(this_lti, displaystr = True):
    
    num_str_aux = build_poly_str(this_lti.num)
    den_str_aux = build_poly_str(this_lti.den)

    strout = r'\frac{' + num_str_aux + '}{' + den_str_aux + '}'

    if displaystr:
        display(Math(strout))
    else:
        return strout

        

def pretty_print_bicuad_omegayq(num, den = None, displaystr = True):
    
    if den is None:
        this_sos = num.reshape((1,6))
    else:
        this_sos = np.hstack((
            np.pad(num, (3-len(num),0)),
            np.pad(den, (3-len(den),0)))
        ).reshape((1,6))
    
    num = this_sos[0,:3]
    den = this_sos[0,3:]
    
    if np.all( np.abs(num) > 0):
        # complete 2nd order, omega and Q parametrization
        num_str_aux = build_omegayq_str(num)
    elif np.all(num[[0,2]] == 0) and num[1] > 0 :
        # bandpass style  s . k = s . H . omega/Q 
        num_str_aux = build_omegayq_str(num, den = den)
    else:
        num_str_aux = build_poly_str(num)
        
    
    den_str_aux = build_omegayq_str(den)
    
    strout = r'\frac{' + num_str_aux + '}{' + den_str_aux + '}'

    if displaystr:
        display(Math(strout))
    else:   
        return strout

def one_sos2tf(mySOS):
    
    # check zeros in the higher order coerffs
    if mySOS[0] == 0 and mySOS[1] == 0:
        num = mySOS[2]
    elif mySOS[0] == 0:
        num = mySOS[1:3]
    else:
        num = mySOS[:3]
        
    if mySOS[3] == 0 and mySOS[4] == 0:
        den = mySOS[-1]
    elif mySOS[3] == 0:
        den = mySOS[4:]
    else:
        den = mySOS[3:]
    
    return num, den


def pretty_print_SOS(mySOS, mode = 'default', displaystr = True):
    '''
    Los SOS siempre deben definirse como:
        
        
        mySOS= ( [ a1_1 a2_1 a3_1 b1_1 b2_1 b3_1 ]
                 [ a1_2 a2_2 a3_2 b1_2 b2_2 b3_2 ]
                 ...
                 [ a1_N a2_N a3_N b1_N b2_N b3_N ]
                )
        
        siendo:
            
                s² a1_i + s a2_i + a3_i
        T_i =  -------------------------
                s² b1_i + s b2_i + b3_i

    Parameters
    ----------
    mySOS : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'default'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    sos_str = '' 
    
    valid_modes = ['default', 'omegayq']
    if mode not in valid_modes:
        raise ValueError('mode must be one of %s, not %s'
                         % (valid_modes, mode))
    SOSnumber, _ = mySOS.shape
    
    for ii in range(SOSnumber):
        
        if mode == "omegayq" and mySOS[ii,3] > 0:
            sos_str += r' . ' + pretty_print_bicuad_omegayq(mySOS[ii,:], displaystr = False )
        else:
            num, den = one_sos2tf(mySOS[ii,:])
            this_tf = TransferFunction(num, den)
            sos_str += r' . ' + pretty_print_lti(this_tf, displaystr = False)

    sos_str = sos_str[2:]

    if displaystr:
        display(Math( r' ' + sos_str))
    else:
        return sos_str



def analyze_sys( all_sys, sys_name = None, img_ext = 'none', same_figs=True ):
    
    
    
    valid_ext = ['none', 'png', 'svg']
    if img_ext not in valid_ext:
        raise ValueError('Image extension must be one of %s, not %s'
                         % (valid_ext, img_ext))
    
    
    if isinstance(all_sys, list):
        cant_sys = len(all_sys)
    else:
        all_sys = [all_sys]
        cant_sys = 1

    if sys_name is None:
        sys_name = [str(ii) for ii in range(cant_sys)]
        
    if not isinstance(sys_name, list):
        sys_name = [sys_name]
        
    ## BODE plots
    if same_figs:
        fig_id = 1
    else:
        fig_id = 'none'
    axes_hdl = ()

    for ii in range(cant_sys):
        fig_id, axes_hdl = bodePlot(all_sys[ii], fig_id, axes_hdl, label = sys_name[ii])

    if img_ext != 'none':
        plt.savefig('_'.join(sys_name) + '_Bode.' + img_ext, format=img_ext)

    # fig_id = 6
    # axes_hdl = ()

    # for ii in range(cant_sys):
    #     fig_id, axes_hdl = bodePlot(all_sys[ii], fig_id, axes_hdl, label = sys_name[ii])

    # axes_hdl[0].set_ylim(bottom=-3)

    # if img_ext != 'none':
    #     plt.savefig('_'.join(sys_name) + '_Bode-3db.' + img_ext, format=img_ext)


    ## PZ Maps
    if same_figs:
        analog_fig_id = 3
        digital_fig_id = 4
    else:
        analog_fig_id = 'none'
        digital_fig_id = 'none'
    
    analog_axes_hdl = ()
    digital_axes_hdl = ()
    
    for ii in range(cant_sys):
    
        if isinstance(all_sys[ii], np.ndarray):
            
            thisFilter = sos2tf_analog(all_sys[ii])

            analog_fig_id, analog_axes_hdl = pzmap(thisFilter, filter_description=sys_name[ii], fig_id = analog_fig_id, axes_hdl=analog_axes_hdl, annotations = True)
            
        else:
                
            if all_sys[ii].dt is None:
                analog_fig_id, analog_axes_hdl = pzmap(all_sys[ii], filter_description=sys_name[ii], fig_id = analog_fig_id, axes_hdl=analog_axes_hdl)
            else:
                digital_fig_id, digital_axes_hdl = pzmap(all_sys[ii], filter_description=sys_name[ii], fig_id = digital_fig_id, axes_hdl=digital_axes_hdl)
            

    if isinstance(all_sys[ii], np.ndarray) or ( isinstance(all_sys[ii], TransferFunction) and all_sys[ii].dt is None) :
        analog_axes_hdl.legend()
        if img_ext != 'none':
            plt.figure(analog_fig_id)
            plt.savefig('_'.join(sys_name) + '_Analog_PZmap.' + img_ext, format=img_ext)
    else:
        digital_axes_hdl.legend()
        if img_ext != 'none':
            plt.figure(digital_fig_id)
            plt.savefig('_'.join(sys_name) + '_Digital_PZmap.' + img_ext, format=img_ext)
    
#    plt.show()
    
    ## Group delay plots
    if same_figs:
        fig_id = 5
    else:
        fig_id = 'none'
    
    for ii in range(cant_sys):
        fig_id, axes_hdl = GroupDelay(all_sys[ii], fig_id, label = sys_name[ii])
    
    # axes_hdl.legend(sys_name)

    axes_hdl.set_ylim(bottom=0)

    if img_ext != 'none':
        plt.savefig('_'.join(sys_name) + '_GroupDelay.'  + img_ext, format=img_ext)



def pzmap(myFilter, annotations = False, filter_description='none', fig_id='none', axes_hdl='none'):
    """Plot the complex s-plane given zeros and poles.
    Pamams:
     - b: array_like. Numerator polynomial coefficients.
     - a: array_like. Denominator polynomial coefficients.
    
    http://www.ehu.eus/Procesadodesenales/tema6/102.html
    
    """

    if fig_id == 'none':
        fig_hdl = plt.figure()
        fig_id = fig_hdl.number
    else:
        if plt.fignum_exists(fig_id):
            fig_hdl = plt.figure(fig_id)
        else:
            fig_hdl = plt.figure(fig_id)
            fig_id = fig_hdl.number

    axes_hdl = plt.gca()
    
        # Get the poles and zeros
    z, p, k = tf2zpk(myFilter.num, myFilter.den)


    # Add unit circle and zero axes    
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='gray', ls='dotted', lw = 2)
    axes_hdl.add_patch(unit_circle)
    plt.axvline(0, color='0.7')
    plt.axhline(0, color='0.7')

    
    #Add circle lines
    
#        maxRadius = np.abs(10*np.sqrt(p[0]))
    
    # Plot the poles and set marker properties
    poles = plt.plot(p.real, p.imag, 'x', markersize=9, label=filter_description)
    
#    if filter_description != 'none':
#        poles[0].label = filter_description
    
    # Plot the zeros and set marker properties
    zeros = plt.plot(z.real, z.imag,  'o', markersize=9, 
             color='none',
             markeredgecolor=poles[0].get_color(), # same color as poles
             markerfacecolor='white'
             )

    # add info to poles and zeros
    # first with poles
    w0, aux_idx = np.unique(np.abs(p), return_index=True)
    qq = 1 / (2*np.cos(np.pi - np.angle(p[aux_idx])))
   
    for ii in range(len(w0)):

        if p[aux_idx[ii]].imag > 0.0:
            # annotate with Q only complex conj singularities
            
            circle = patches.Circle((0,0), radius=w0[ii], color = poles[0].get_color(), fill=False, ls= (0, (1, 10)), lw = 0.7)
            
            axes_hdl.add_patch(circle)
            plt.axvline(0, color='0.7')
            plt.axhline(0, color='0.7')
    
            if annotations:
                axes_hdl.annotate('$\omega$ = {:3.3g} \n Q = {:3.3g}'.format(w0[ii], qq[ii]),
                            xy=(p[aux_idx[ii]].real, p[aux_idx[ii]].imag), xycoords='data',
                            xytext=(-25, 30), textcoords='offset points',
                            arrowprops=dict(facecolor='black', shrink=0.15,
                                            width = 1, headwidth = 5 ),
                            horizontalalignment='right', verticalalignment='bottom')
    
        else:
            # annotate with omega real singularities
            
            if annotations:
                axes_hdl.annotate('$\omega$ = {:3.3g}'.format(w0[ii]),
                            xy=(p[aux_idx[ii]].real, p[aux_idx[ii]].imag), xycoords='data',
                            xytext=(-25, 30), textcoords='offset points',
                            arrowprops=dict(facecolor='black', shrink=0.15,
                                            width = 1, headwidth = 5 ),
                            horizontalalignment='right', verticalalignment='bottom')
            

    # and then zeros
    w0, aux_idx = np.unique(np.abs(z), return_index=True)
    qq = 1 / (2*np.cos(np.pi - np.angle(z[aux_idx])))

    for ii in range(len(w0)):

        if z[aux_idx[ii]].imag > 0.0:
            
            circle = patches.Circle((0,0), radius=w0[ii], color = poles[0].get_color(), fill=False, ls= (0, (1, 10)), lw = 0.7)
            
            axes_hdl.add_patch(circle)
            plt.axvline(0, color='0.7')
            plt.axhline(0, color='0.7')
    
            if annotations:
                axes_hdl.annotate('$\omega$ = {:3.3g} \n Q = {:3.3g}'.format(w0[ii], qq[ii]),
                            xy=(z[aux_idx[ii]].real, z[aux_idx[ii]].imag), xycoords='data',
                            xytext=(-25, 30), textcoords='offset points',
                            arrowprops=dict(facecolor='black', shrink=0.15,
                                            width = 1, headwidth = 5 ),
                            horizontalalignment='right', verticalalignment='bottom')
    
        else:
            # annotate with omega real singularities
            
            if annotations:
                axes_hdl.annotate('$\omega$ = {:3.3g}'.format(w0[ii]),
                            xy=(z[aux_idx[ii]].real, z[aux_idx[ii]].imag), xycoords='data',
                            xytext=(-25, 30), textcoords='offset points',
                            arrowprops=dict(facecolor='black', shrink=0.15,
                                            width = 1, headwidth = 5 ),
                            horizontalalignment='right', verticalalignment='bottom')


    # Scale axes to fit
    r_old = axes_hdl.get_ylim()[1]
    
    r = 1.1 * np.amax(np.concatenate(([r_old/1.1], abs(z), abs(p), [1])))
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
#    ticks = [-1, -.5, .5, 1]
#    plt.xticks(ticks)
#    plt.yticks(ticks)

    """
    If there are multiple poles or zeros at the same point, put a 
    superscript next to them.
    TODO: can this be made to self-update when zoomed?
    """
    # Finding duplicates by same pixel coordinates (hacky for now):
    poles_xy = axes_hdl.transData.transform(np.vstack(poles[0].get_data()).T)
    zeros_xy = axes_hdl.transData.transform(np.vstack(zeros[0].get_data()).T)    

    # dict keys should be ints for matching, but coords should be floats for 
    # keeping location of text accurate while zooming

    

    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in poles_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = axes_hdl.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )

    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in zeros_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = axes_hdl.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )

    

    plt.xlabel(r'$\sigma$')
    plt.ylabel('j'+r'$\omega$')

    plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')

    fig_hdl.suptitle('Poles and Zeros map')

    axes_hdl.legend()

    return fig_id, axes_hdl
    

def GroupDelay(myFilter, fig_id='none', label = '', npoints = 1000):

    
    if isinstance(myFilter, np.ndarray):
        # SOS section
        cant_sos = myFilter.shape[0]
        phase = np.empty((npoints, cant_sos+1))
        sos_label = []
        
        for ii in range(cant_sos):
            
            num, den = one_sos2tf(myFilter[ii,:])
            thisFilter = TransferFunction(num, den)
            w, _, phase[:,ii] = thisFilter.bode(np.logspace(-2,2,npoints))
            sos_label += [label + ' - SOS {:d}'.format(ii)]
        
        # whole filter
        thisFilter = sos2tf_analog(myFilter)
        w, _, phase[:,cant_sos] = thisFilter.bode(np.logspace(-2,2,npoints))
        sos_label += [label]
        
        label = sos_label
        
    else:
        # LTI object
        cant_sos = 0
        w,_,phase = myFilter.bode( np.logspace(-2,2,npoints) )
        
        if isinstance(label, str):
            label = [label]


    phaseRad = phase * np.pi / 180.0
    groupDelay = -np.diff(phaseRad.reshape((npoints, 1+cant_sos)), axis = 0)/np.diff(w).reshape((npoints-1,1))

    if fig_id == 'none':
        fig_hdl = plt.figure()
        fig_id = fig_hdl.number
    else:
        if plt.fignum_exists(fig_id):
            fig_hdl = plt.figure(fig_id)
        else:
            fig_hdl = plt.figure(fig_id)
            fig_id = fig_hdl.number

    aux_hdl = plt.semilogx(w[1:], groupDelay, label=label)    # Bode phase plot

    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)
    plt.xlabel('Angular frequency [rad/sec]')
    plt.ylabel('Group Delay [sec]')
    plt.title('Group delay')

    axes_hdl = plt.gca()
    
    if label != '' :
        # axes_hdl.legend( label )
        axes_hdl.legend()

    return fig_id, axes_hdl

def bodePlot(myFilter, fig_id='none', axes_hdl='none', label = '', npoints = 1000 ):
    
    if isinstance(myFilter, np.ndarray):
        # SOS section
        cant_sos = myFilter.shape[0]
        mag = np.empty((npoints, cant_sos+1))
        phase = np.empty_like(mag)
        sos_label = []
        
        for ii in range(cant_sos):
            
            num, den = one_sos2tf(myFilter[ii,:])
            thisFilter = TransferFunction(num, den)
            w, mag[:, ii], phase[:,ii] = thisFilter.bode(np.logspace(-2,2,npoints))
            sos_label += [label + ' - SOS {:d}'.format(ii)]
        
        # whole filter
        thisFilter = sos2tf_analog(myFilter)
        w, mag[:, cant_sos], phase[:,cant_sos] = thisFilter.bode(np.logspace(-2,2,npoints))
        sos_label += [label]
        
        label = sos_label
        
    else:
        # LTI object
        cant_sos = 0
        w, mag, phase = myFilter.bode(np.logspace(-2,2,npoints))
        
        if isinstance(label, str):
            label = [label]
        

    if fig_id == 'none':
        fig_hdl, axes_hdl = plt.subplots(2, 1, sharex='col')
        fig_id = fig_hdl.number
    else:
        if plt.fignum_exists(fig_id):
            fig_hdl = plt.figure(fig_id)
            axes_hdl = fig_hdl.get_axes()
        else:
            fig_hdl = plt.figure(fig_id)
            axes_hdl = fig_hdl.subplots(2, 1, sharex='col')
            fig_id = fig_hdl.number

    (mag_ax_hdl, phase_ax_hdl) = axes_hdl
    
    plt.sca(mag_ax_hdl)
    aux_hdl = plt.semilogx(w, mag, label=label)    # Bode magnitude plot
    
    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)
#    plt.xlabel('Angular frequency [rad/sec]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Magnitude response')
    
    if label != '' :
        # mag_ax_hdl.legend( label )
        mag_ax_hdl.legend()

        
    plt.sca(phase_ax_hdl)
    aux_hdl = plt.semilogx(w, phase, label=label)    # Bode phase plot
    
    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)
    plt.xlabel('Angular frequency [rad/sec]')
    plt.ylabel('Phase [deg]')
    plt.title('Phase response')
    
    if label != '' :
        # phase_ax_hdl.legend( label )
        phase_ax_hdl.legend()
    
    return fig_id, axes_hdl
    
def sos2tf_analog(mySOS):
    
    SOSnumber, _ = mySOS.shape
    
    num = 1
    den = 1
    
    for ii in range(SOSnumber):
        
        sos_num, sos_den = one_sos2tf(mySOS[ii,:])
        num = np.polymul(num, sos_num)
        den = np.polymul(den, sos_den)

    tf = TransferFunction(num, den)
    
    return tf

def tf2sos_analog(num, den, pairing='nearest'):

    z, p, k = tf2zpk(num, den)
    
    sos = zpk2sos_analog(z, p, k, pairing = pairing)

    return sos
        
def zpk2sos_analog(z, p, k, pairing='nearest'):
    """
    From scipy.signal, modified by marianux
    ----------------------------------------
    
    Return second-order sections from zeros, poles, and gain of a system
    
    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {'nearest', 'keep_odd'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See Notes below.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt

    Notes
    -----
    The algorithm used to convert ZPK to SOS format follows the suggestions
    from R. Schaumann's "Design of analog filters". Ch. 5:
        1- Assign zeros to closest poles
        2- order sections by increasing Q
        3- gains ordering to maximize dynamic range. See ch. 5.

  
    """
    
    # if empty filter then
    if len(z) == len(p) == 0:
        return np.array([[0., 0., k, 1., 0., 0.]])

    assert len(z) <= len(p), "Filter must have more poles than zeros"
    
    n_sections = ( len(p) + 1) // 2
    sos = np.zeros((n_sections, 6))

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))

    # calculate los omega_0 and Q for each pole
    # w0 = np.abs(p)
    qq = 1 / (2*np.cos(np.pi - np.angle(p)))

    p_sos = np.zeros((n_sections, 2), np.complex128)
    z_sos = np.zeros_like(p_sos)
    
    if n_sections == z.shape[0]:
        one_z_per_section = True
    else:
        one_z_per_section = False
            
    
    for si in range(n_sections):
        # Select the next "worst" pole
        p1_idx = np.argmax(qq)
            
        p1 = p[p1_idx]
        p = np.delete(p, p1_idx)
        qq = np.delete(qq, p1_idx)

        # Pair that pole with a zero

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            # Special case to set a first-order section
            if z.size == 0:
                # no zero, just poles
                z1 = np.nan

            else:            
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)
                
            p2 = z2 = np.nan
            
        else:
            
            if z.size == 0:
                # no zero, just poles
                z1 = np.nan
                
            else:
                # Pair the pole with the closest zero (real or complex)
                z1_idx = np.argmin(np.abs(p1 - z))
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)

            # Now that we have p1 and z1, figure out what p2 and z2 need to be
            
            if z.size == 0:
                # no zero, just poles
                if np.isreal(p1):
                    # pick the next "worst" pole to use
                    idx = np.nonzero(np.isreal(p))[0]
                    assert len(idx) > 0
                    p2_idx = idx[np.argmax(qq)]
                    p2 = p[p2_idx]
                    z2 = np.nan
                    p = np.delete(p, p2_idx)

                else:
                    # complex pole
                    p2 = p1.conj()
                    z2 = np.nan
                
            else:
                # there are zero/s for z2
                    
                if not np.isreal(p1):
                    p2 = p1.conj()
                    
                    if not np.isreal(z1):  # complex pole, complex zero
                        z2 = z1.conj()
                    else:  # complex pole, real zero
                        
                        if one_z_per_section:
                            # avoid picking double zero (high-pass)
                            # prefer picking band-pass sections (Schaumann 5.3.1)
                            z2 = np.nan
                        else:
                            z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                            z2 = z[z2_idx]
                            assert np.isreal(z2)
                            z = np.delete(z, z2_idx)
                else:
                    if not np.isreal(z1):  # real pole, complex zero
                        z2 = z1.conj()
                        p2_idx = _nearest_real_complex_idx(p, z1, 'real')
                        p2 = p[p2_idx]
                        assert np.isreal(p2)
                    else:  # real pole, real zero
                        # pick the next "worst" pole to use
                        idx = np.nonzero(np.isreal(p))[0]
                        assert len(idx) > 0
                        p2_idx = idx[np.argmin(np.abs(np.abs(p[idx]) - 1))]
                        p2 = p[p2_idx]
                        # find a real zero to match the added pole
                        assert np.isreal(p2)
                        
                        if one_z_per_section:
                            # avoid picking double zero (high-pass)
                            # prefer picking band-pass sections (Schaumann 5.3.1)
                            z2 = np.nan
                        else:
                            z2_idx = _nearest_real_complex_idx(z, p2, 'real')
                            z2 = z[z2_idx]
                            assert np.isreal(z2)
                            z = np.delete(z, z2_idx)
                    p = np.delete(p, p2_idx)
                    
        p_sos[si] = [p1, p2]
        z_sos[si] = [z1, z2]
        
    assert len(p) == 0  # we've consumed all poles and zeros
    del p, z

    # Construct the system, reversing order so the "worst" are last
    p_sos = np.reshape(p_sos[::-1], (n_sections, 2))
    z_sos = np.reshape(z_sos[::-1], (n_sections, 2))
    
    maxima_tf = np.ones(n_sections)
    gains = np.ones(n_sections, np.array(k).dtype)
    # gains[0] = k # todo: distribute k along sections
    
    for si in range(n_sections):
        
        num, den = zpk2tf(z_sos[si, np.logical_not( np.isnan(z_sos[si])) ], p_sos[si, np.logical_not(np.isnan(p_sos[si]))], 1) # no gain
        
        # find maximum in transfer function
        thisFilter = TransferFunction(num, den)
        
        _, mag, _ = thisFilter.bode(np.logspace(-2,2,100))
        
        maxima_tf[si] = np.max(mag)
    
    mmi = np.cumprod(maxima_tf) # M_i according to Schaumann eq 5.76

    # first gain to optimize dynamic range.
    gains[0] = k * (mmi[-1]/mmi[0])

    for si in range(n_sections):

        if si > 0:
            gains[si] = (mmi[si-1]/mmi[si])

        num, den = zpk2tf(z_sos[si, np.logical_not(np.isnan(z_sos[si])) ], p_sos[si, np.logical_not(np.isnan(p_sos[si]))], gains[si]) # now with gain
        
        num = np.concatenate((np.zeros(np.max(3 - len(num), 0)), num))
        den = np.concatenate((np.zeros(np.max(3 - len(den), 0)), den))
            
        sos[si] = np.concatenate((num,den))
        
    return sos
    
    # SOSarray = tf2sos(myFilter.num, myFilter.den)
    
    # SOSnumber,_ = SOSarray.shape
    
    # SOSoutput = np.empty(shape=(SOSnumber,3))
    
    # for index in range(SOSnumber):
    #     SOSoutput[index][:] = SOSarray[index][3::]
        
    #     if SOSoutput[index][2]==0:
    #         SOSoutput[index] = np.roll(SOSoutput[index],1)
        
    # return SOSoutput

def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex')
    order = np.argsort(np.abs(fro - to))
    mask = np.isreal(fro[order])
    if which == 'complex':
        mask = ~mask
    return order[np.nonzero(mask)[0][0]]

def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (`zc`) and real (`zr`)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxpair

    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    z = np.atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Get tolerance from dtype of input
        tol = 100 * np.finfo((1.0 * z).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[np.lexsort((abs(z.imag), z.real))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return np.array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(np.concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr
