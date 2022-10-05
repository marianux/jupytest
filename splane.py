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
from sympy.abc import s
from sympy.polys.partfrac import apart

from schemdraw import Drawing
from schemdraw.elements import  Resistor, ResistorIEC, Capacitor, Inductor, Line, Dot, Gap, Arrow

from fractions import Fraction

s = sp.symbols('s', complex=True)


def dibujar_cauer_RC_RL(ki = None, y_exc = None, z_exc = None):
    '''
    Description
    -----------
    Draws a parallel non-disipative admitance following Foster synthesis method.

        YorZ = ki / s +  1 / ( ki_i / s + koo_i * s ) 
    
    Parameters
    ----------
    ki : symbolic positive real number. The residue value at DC or s->0.
        
    koo : symbolic positive real number. The residue value at inf or s->oo.
        
    ki : symbolic positive real array of numbers. A list of residue pairs at 
         each i-th finite pole or s**2->-(w_i**2). The first element of the pair
         is the ki_i value (capacitor), while the other is the koo_i (inductor)
         value.

    Returns
    -------
    The drawing object.
    
    Ejemplo
    -------

    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Foster
    ki, koo, ki = tc2.foster(Imm)
    
    # Tratamos a nuestra función imitancia como una Z
    tc2.dibujar_foster_derivacion(ki, koo, ki, y_exc = Imm)

    '''    
    if y_exc is None and z_exc is None:

        assert('Hay que definir si se trata de una impedancia o admitancia')

    if not(ki is None) or len(ki) > 0:
        # si hay algo para dibujar ...
        
        d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

        d = dibujar_puerto_entrada(d,
                                       voltage_lbl = ('+', '$V$', '-'), 
                                       current_lbl = '$I$')

        if y_exc is None:
            
            bIsImpedance = True
            
            d, _ = dibujar_funcion_exc_abajo(d, 
                                                      'Z',  
                                                      z_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)
        else:
            bIsImpedance = False
            
            d, _ = dibujar_funcion_exc_abajo(d, 
                                                      'Y',  
                                                      y_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)
    
        if bIsImpedance:
            bSeries = True
        else:
            bSeries = False
        
        bComponenteDibujadoDerivacion = False

        for kii in ki:


            if bSeries:
                
                if sp.degree(kii*s) == 1:
                    d = dibujar_elemento_serie(d, Resistor, kii)
                elif sp.degree(kii*s) == 0:
                    d = dibujar_elemento_serie(d, Capacitor, 1/(s*kii))
                else:
                    d = dibujar_elemento_serie(d, Inductor, kii/s)
                    
                bComponenteDibujadoDerivacion = False

            else:

                if bComponenteDibujadoDerivacion:
                    
                    dibujar_espacio_derivacion(d)

                if sp.degree(kii*s) == 1:
                    d = dibujar_elemento_derivacion(d, Resistor, 1/kii)
                elif sp.degree(kii*s) == 2:
                    d = dibujar_elemento_derivacion(d, Capacitor, kii/s)
                else:
                    d = dibujar_elemento_derivacion(d, Inductor, 1/(s*kii))
                
                bComponenteDibujadoDerivacion = True

            bSeries = not bSeries

        if not bComponenteDibujadoDerivacion:
            
            d += Line().right().length(d.unit*.25)
            d += Line().down()
            d += Line().left().length(d.unit*.25)
        
        display(d)

    else:    
        
        print('Nada para dibujar')


def dibujar_cauer_LC(ki = None, y_exc = None, z_exc = None):
    '''
    Description
    -----------
    Draws a parallel non-disipative admitance following Foster synthesis method.

        YorZ = ki / s +  1 / ( ki_i / s + koo_i * s ) 
    
    Parameters
    ----------
    ki : symbolic positive real number. The residue value at DC or s->0.
        
    koo : symbolic positive real number. The residue value at inf or s->oo.
        
    ki : symbolic positive real array of numbers. A list of residue pairs at 
         each i-th finite pole or s**2->-(w_i**2). The first element of the pair
         is the ki_i value (capacitor), while the other is the koo_i (inductor)
         value.

    Returns
    -------
    The drawing object.
    
    Ejemplo
    -------

    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Foster
    ki, koo, ki = tc2.foster(Imm)
    
    # Tratamos a nuestra función imitancia como una Z
    tc2.dibujar_foster_derivacion(ki, koo, ki, y_exc = Imm)

    '''    
    if y_exc is None and z_exc is None:

        assert('Hay que definir si se trata de una impedancia o admitancia')

    if not(ki is None) or len(ki) > 0:
        # si hay algo para dibujar ...
        
        d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

        d = dibujar_puerto_entrada(d,
                                       voltage_lbl = ('+', '$V$', '-'), 
                                       current_lbl = '$I$')

        if y_exc is None:
            
            bIsImpedance = True
            
            d, _ = dibujar_funcion_exc_abajo(d, 
                                                      'Z',  
                                                      z_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)
        else:
            bIsImpedance = False
            
            d, _ = dibujar_funcion_exc_abajo(d, 
                                                      'Y',  
                                                      y_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)
    
        if bIsImpedance:
            bSeries = True
        else:
            bSeries = False

        # 1/s me da orden 1, atenti.
        if sp.degree(ki[0]*s) == 2 :
            bCauer1 = True
        else:
            bCauer1 = False
        
        
        bComponenteDibujadoDerivacion = False

        for kii in ki:


            if bSeries:
                
                if bCauer1:
                    d = dibujar_elemento_serie(d, Inductor, kii/s)
                else:
                    d = dibujar_elemento_serie(d, Capacitor, 1/(s*kii))
                    
                bComponenteDibujadoDerivacion = False

            else:

                if bComponenteDibujadoDerivacion:
                    
                    dibujar_espacio_derivacion(d)

                if bCauer1:
                    d = dibujar_elemento_derivacion(d, Capacitor, kii/s)
                else:
                    d = dibujar_elemento_derivacion(d, Inductor, 1/(s*kii))
                
                bComponenteDibujadoDerivacion = True

            bSeries = not bSeries

        if not bComponenteDibujadoDerivacion:
            
            d += Line().right().length(d.unit*.25)
            d += Line().down()
            d += Line().left().length(d.unit*.25)
        
        display(d)

    else:    
        
        print('Nada para dibujar')



def cauer_RC( imm, remover_en_inf=True ):
    '''
    Description
    -----------
    Perform continued fraction expansion over imm following Cauer 2 synthesis method.

        imm = k0_0 / s + 1 / ( k0_1 + 1/ (k0_2 / s  + 1/ ... )) 

    Parameters
    ----------
    immittance : symbolic rational function
        La inmitancia a sintetizar.

    Returns
    -------
    A list k0 with the i-th k0_i resulted from continued fraction expansion.

    Ejemplo
    -------
    
    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Cauer 1 o remociones continuas en infinito
    imm_cauer_0, k0 = tc2.cauer_0(Imm)

    '''    
    
    ko = []

    if remover_en_inf:
        rem, koi = remover_polo_infinito(imm)
        bRemoverPolo = False

        if koi.is_zero:
            rem, koi = remover_valor_en_infinito(imm)
            bRemoverPolo = True
            
    else:
        
        rem, koi = remover_polo_dc(imm)
        bRemoverPolo = False

        if koi.is_zero:
            rem, koi = remover_valor_en_dc(imm)
            bRemoverPolo = True

    
        
    while not(rem.is_zero) and not(koi.is_zero):
        
        ko += [koi]
        rem = 1/rem

        if remover_en_inf:
            
            if bRemoverPolo:
                rem, koi = remover_polo_infinito(rem)
                bRemoverPolo = False
            else:
                rem, koi = remover_valor_en_infinito(rem)
                bRemoverPolo = True
        else:
            
            if bRemoverPolo:
                rem, koi = remover_polo_dc(rem)
                bRemoverPolo = False
            else:
                rem, koi = remover_valor_en_dc(rem)
                bRemoverPolo = True


    if koi.is_zero:
        # deshago para entender al resto de la misma 
        # naturaleza que el último elemento que retiró.
        rem = 1/rem
    else:
        ko += [koi]

    imm_as_cauer = koi
    
    for ii in np.flipud(np.arange(len(ko)-1)):

        imm_as_cauer = ko[ii] + 1/imm_as_cauer
        
    return(ko, imm_as_cauer, rem)

def cauer_LC( imm, remover_en_inf = True ):
    '''
    Description
    -----------
    Perform continued fraction expansion over imm following Cauer 1 synthesis method.

        imm = koo_0 * s + 1 / ( koo_1 * s + 1/ (koo_2 * s  + 1/ ... )) 

    Parameters
    ----------
    immittance : symbolic rational function
        La inmitancia a sintetizar.

    Returns
    -------
    A list koo with the i-th koo_i resulted from continued fraction expansion.

    Ejemplo
    -------
    
    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Cauer 1 o remociones continuas en infinito
    imm_cauer_oo, koo = tc2.cauer_oo(Imm)

    '''    
        
    rem = imm
    ko = []

    if remover_en_inf:
        rem, koi = remover_polo_infinito(rem)
    else:
        rem, koi = remover_polo_dc(rem)
        
    
    while not(rem.is_zero) and not(koi.is_zero):
        
        ko += [koi]
        rem = 1/rem

        if remover_en_inf:
            rem, koi = remover_polo_infinito(rem)
        else:
            rem, koi = remover_polo_dc(rem)

    if koi.is_zero:
        # deshago para entender al resto de la misma 
        # naturaleza que el último elemento que retiró.
        rem = 1/rem
    else:
        ko += [koi]

    imm_as_cauer = koi

    for ii in np.flipud(np.arange(len(ko)-1)):
        
        imm_as_cauer = ko[ii] + 1/imm_as_cauer
        
    return(ko, imm_as_cauer, rem)



def dibujar_foster_derivacion(k0 = None, koo = None, ki = None, y_exc = None):
    '''
    Description
    -----------
    Draws a parallel non-disipative admitance following Foster synthesis method.

        Y = k0 / s + koo * s +  1 / ( k0_i / s + koo_i * s ) 
    
    Parameters
    ----------
    k0 : symbolic positive real number. The residue value at DC or s->0.
        
    koo : symbolic positive real number. The residue value at inf or s->oo.
        
    ki : symbolic positive real array of numbers. A list of residue pairs at 
         each i-th finite pole or s**2->-(w_i**2). The first element of the pair
         is the k0_i value (capacitor), while the other is the koo_i (inductor)
         value.

    Returns
    -------
    The drawing object.
    
    Ejemplo
    -------

    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Foster
    k0, koo, ki = tc2.foster(Imm)
    
    # Tratamos a nuestra función imitancia como una Z
    tc2.dibujar_foster_derivacion(k0, koo, ki, y_exc = Imm)

    '''    

    if not(k0 is None and koo is None and ki is None):
        # si hay algo para dibujar ...
        
        d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

        bComponenteDibujado = False

        d = dibujar_puerto_entrada(d,
                                       voltage_lbl = ('+', '$V$', '-'), 
                                       current_lbl = '$I$')

        if not(y_exc is None):
            d, _ = dibujar_funcion_exc_abajo(d, 
                                                      'Y',  
                                                      y_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)

        if not(k0 is None):
        
            d = dibujar_elemento_derivacion(d, Inductor, 1/k0)
            
            bComponenteDibujado = True
            
            
        if not(koo is None):
        
            if bComponenteDibujado:
                
                dibujar_espacio_derivacion(d)
                    
            d = dibujar_elemento_derivacion(d, Capacitor, koo)

            bComponenteDibujado = True
            
        if not(ki is None):

            for un_tanque in ki:

                if bComponenteDibujado:
                    
                    dibujar_espacio_derivacion(d)
                
                d = dibujar_tanque_derivacion(d, inductor_lbl = un_tanque[1], capacitor_lbl = 1/un_tanque[0])

                bComponenteDibujado = True

        
        display(d)

    else:    
        
        print('Nada para dibujar')


def dibujar_foster_serie(k0 = None, koo = None, ki = None, z_exc = None):
    '''
    Description
    -----------
    Draws a series non-disipative impedance following Foster synthesis method.

        Z = k0 / s + koo * s +  1 / ( k0_i / s + koo_i * s ) 
    
    Parameters
    ----------
    k0 : symbolic positive real number. The residue value at DC or s->0.
        
    koo : symbolic positive real number. The residue value at inf or s->oo.
        
    ki : symbolic positive real array of numbers. A list of residue pairs at 
         each i-th finite pole or s**2->-(w_i**2). The first element of the pair
         is the k0_i value (inductor), while the other is the koo_i (capacitor)
         value.

    Returns
    -------
    The drawing object.
    
    Ejemplo
    -------

    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Foster
    k0, koo, ki = tc2.foster(Imm)
    
    # Tratamos a nuestra función imitancia como una Z
    tc2.dibujar_foster_serie(k0, koo, ki, z_exc = Imm)

    '''    

    if not(k0 is None and koo is None and ki is None):
        # si hay algo para dibujar ...
        
        d = Drawing(unit=4)  # unit=2 makes elements have shorter than normal leads

        d = dibujar_puerto_entrada(d,
                                       voltage_lbl = ('+', '$V$', '-'), 
                                       current_lbl = '$I$')

        if not(z_exc is None):
            d, z5_lbl = dibujar_funcion_exc_abajo(d, 
                                                      'Z',  
                                                      z_exc, 
                                                      hacia_salida = True,
                                                      k_gap_width = 0.5)

        if not(k0 is None):
        
            d = dibujar_elemento_serie(d, Capacitor, 1/k0)
            
        if not(koo is None):
        
            d = dibujar_elemento_serie(d, Inductor, koo)
            
        if not(ki is None):

            for un_tanque in ki:
                
                d = dibujar_tanque_serie(d, inductor_lbl = 1/un_tanque[0], capacitor_lbl = un_tanque[1] )

                dibujar_espacio_derivacion(d)


        d += Line().right().length(d.unit*.25)
        d += Line().down()
        d += Line().left().length(d.unit*.25)
        
        display(d)
        
        return(d)

    else:    
        
        print('Nada para dibujar')



def foster( imm ):
    '''
    Parameters
    ----------
    immittance : symbolic rational function
        La inmitancia a sintetizar.

    Returns
    -------
    Una lista imm_list con los elementos obtenidos de la siguientes expansión en 
    fracciones simples:
        
        Imm = k0 / s + koo * s +  1 / ( k0_i / s + koo_i * s ) 


    imm_list = [ k0, koo, [k00, koo0], [k01, koo1], ..., [k0N, kooN]  ]
    
    Si algún elemento no está presente, su valor será de "None".

    Ejemplo
    -------
    
    # Sea la siguiente función de excitación
    Imm = (2*s**4 + 20*s**2 + 18)/(s**3 + 4*s)
    
    # Implementaremos Imm mediante Foster
    k0, koo, ki = tc2.foster(Imm)


    '''    
        
    imm_foster = apart(imm)
    
    all_terms = imm_foster.as_ordered_terms()
    
    k0 = None
    koo = None
    ki = []
    ii = 0
    
    for this_term in all_terms:
        
        num, den = this_term.as_numer_denom()
        
        if sp.degree(num) == 1 and sp.degree(den) == 0:
        
            koo = num.as_poly().LC() / den
            
        elif sp.degree(den) == 1 and sp.degree(num) == 0:
            
            k0 = den.as_poly().LC() / num
    
        elif sp.degree(num) == 1 and sp.degree(den) == 2:
            # tanque
            tank_el = (den / num).expand().as_ordered_terms()
    
            koo_i = None
            k0_i = None
            
            for this_el in tank_el:
                
                num, den = this_el.as_numer_denom()
                
                if sp.degree(num) == 1 and sp.degree(den) == 0:
                
                    koo_i = num.as_poly().LC() / den
    
                elif sp.degree(den) == 1 and sp.degree(num) == 0:
                    
                    k0_i = num / den.as_poly().LC() 
                    
            
            ki += [[k0_i, koo_i]]
            ii += 1
            
        else:
            # error
            assert('Error al expandir en fracciones simples.')
    
    if ii == 0:
        ki = None

    return([k0, koo, ki])




def parametrize_sos(num, den):
    
    '''
    Parameters
    ----------
    num : TYPE
        DESCRIPTION.
    den : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Example
    -------

    num = sp.Poly((a*s + b),s)
    den = sp.Poly((c*s + d),s)
    sos_bili, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((a*s),s)
    sos_bili1, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((a),s)
    sos_bili2, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((a*s**2 + b*s + c),s)
    den = sp.Poly((d*s**2 + e*s + f),s)
    sos_1, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((a*s**2 + c**2),s)
    sos_2, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((a*s**2 + s*b),s)
    sos_3, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly(a,s)
    sos_4, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly(a*s**2 ,s)
    sos_5, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    num = sp.Poly((b*s),s)
    sos_6, w_on, Q_n, w_od, Q_d, K = parametrize_sos(num, den)

    '''    
    
    w_od = sp.Rational('0')
    Q_d = sp.Rational('0')
    w_on = sp.Rational('0')
    Q_n = sp.Rational('0')
    K = sp.Rational('0')
    
    den_coeffs = den.all_coeffs()
    num_coeffs = num.all_coeffs()

    if len(den_coeffs) == 3:
    # only 2nd order denominators allowed
        
        w_od = sp.sqrt(den_coeffs[2]/den_coeffs[0])
        
        omega_Q = den_coeffs[1]/den_coeffs[0]
        
        Q_d = sp.simplify(sp.expand(w_od / omega_Q))
        
        k_d = den_coeffs[0]
        
        # wo-Q parametrization
        den  = sp.poly( s**2 + s * sp.Mul(w_od, 1/Q_d, evaluate=False) + w_od**2, s)


        if num.is_monomial:
            
            if num.degree() == 2:
                #pasaaltos
                
                k_n = num_coeffs[0]
                
                num  = sp.poly( s**2, s)

            elif num.degree() == 1:
                #pasabanda
                
                k_n = num_coeffs[0] * Q_d / w_od
                
                # wo-Q parametrization
                num  = sp.poly( s * w_od / Q_d , s)

            else:
                #pasabajos
                
                k_n = num_coeffs[0] / w_od**2
                
                num  = sp.poly( w_od**2, s)

                
        else:
        # no monomial
        
            if num.degree() == 2:

                if num_coeffs[1].is_zero:
                    
                    # zero at w_on
                    w_on = sp.sqrt(num_coeffs[2]/num_coeffs[0])

                    k_n = num_coeffs[0]
                
                    num  = sp.poly( s**2 + w_on**2, s)

                if num_coeffs[2].is_zero:
                
                    # zero at w=0 and at w_on
                    w_on = num_coeffs[1]/num_coeffs[0]

                    k_n = num_coeffs[0]

                    num = sp.poly( s*( s + w_on), s)
                
                else: 
                    # complete poly -> full bicuad
                
                    w_on = sp.sqrt(num_coeffs[2]/num_coeffs[0])
                
                    omega_Q = num_coeffs[1]/num_coeffs[0]
                    
                    Q_n = sp.simplify(sp.expand(w_on / omega_Q))
                    
                    k_n = num_coeffs[0]
                    
                    # wo-Q parametrization
                    num  = sp.poly( s**2 + s * sp.Mul(w_on, 1/Q_n, evaluate=False) + w_on**2, s)

            
            else:
                # only first order
                
                w_on = num_coeffs[1] / num_coeffs[0]
                
                k_n = num_coeffs[0]
                
                num  = sp.poly( s * w_on, s)

        
        K = sp.simplify(sp.expand(k_n / k_d))

    elif len(den_coeffs) == 2:
        # bilineal
        w_od = den_coeffs[1]/den_coeffs[0]
        
        k_d = den_coeffs[0]
        
        # wo-Q parametrization
        den  = sp.poly( s + w_od, s)        
    
        if num.is_monomial:
            
            if num.degree() == 1:
                
                k_n = num_coeffs[0]
                
                # wo-Q parametrization
                num = sp.poly( s, s)        

            else:
                                
                k_n = num_coeffs[0] / w_od
                
                num  = sp.poly( w_od, s)

                
        else:
        # no monomial
        
            w_on = num_coeffs[1]/num_coeffs[0]
            
            k_n = num_coeffs[0]
            
            # wo-Q parametrization
            num = sp.poly( s + w_on, s)        
    
        K = sp.simplify(sp.expand(k_n / k_d))

    return( num, den, w_on, Q_n, w_od, Q_d, K )


def simplify_n_monic(tt):
    
    num, den = sp.fraction(tt)
    
    num = sp.poly(num,s)
    den = sp.poly(den,s)
    
    lcnum = sp.LC(num)
    lcden = sp.LC(den)
    
    k = num.LC() / den.LC()
    
    num = num.monic()
    den = den.monic()

    return( sp.Mul(k,num/den, evaluate=False) )

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
    d += Gap().down().label('').length(0.5*half_width)
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

def dibujar_espacio_derivacion(d):

    d += Line().right().length(d.unit*.25)
    d.push()
    d += Gap().down().label( '' )
    d += Line().left().length(d.unit*.25)
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


def dibujar_tanque_RC_serie(d, sym_R_label='', capacitor_lbl=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(capacitor_lbl, sp.Number ):
        capacitor_lbl = to_latex(capacitor_lbl)
    else:
        capacitor_lbl = str_to_latex(capacitor_lbl)
    
    d.push()
    d += Dot()
    d += Capacitor().right().label(capacitor_lbl, fontsize=16)
    d.pop()
    d += Line().up().length(d.unit*.5)
    d += Resistor().right().label(sym_R_label, fontsize=16)
    d += Line().down().length(d.unit*.5)
    d += Dot()
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_tanque_RC_derivacion(d, sym_R_label='', capacitor_lbl=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(capacitor_lbl, sp.Number ):
        capacitor_lbl = to_latex(capacitor_lbl)
    else:
        capacitor_lbl = str_to_latex(capacitor_lbl)
    
    d.push()
    d += Dot()
    d += Capacitor().down().label(capacitor_lbl, fontsize=16).length(d.unit*.5)
    d += Resistor().down().label(sym_R_label, fontsize=16).length(d.unit*.5)
    d += Dot()
    d.pop()

    return(d)

def dibujar_tanque_RL_serie(d, sym_R_label='', sym_ind_label=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(sym_ind_label, sp.Number ):
        sym_ind_label = to_latex(sym_ind_label)
    else:
        sym_ind_label = str_to_latex(sym_ind_label)
    
    d.push()
    d += Dot()
    d += Inductor().right().label(sym_ind_label, fontsize=16)
    d.pop()
    d += Line().up().length(d.unit*.5)
    d += Resistor().right().label(sym_R_label, fontsize=16)
    d += Line().down().length(d.unit*.5)
    d += Dot()
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_tanque_RL_derivacion(d, sym_R_label='', sym_ind_label=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(sym_ind_label, sp.Number ):
        sym_ind_label = to_latex(sym_ind_label)
    else:
        sym_ind_label = str_to_latex(sym_ind_label)
    
    d.push()
    d += Dot()
    d += Inductor().down().label(sym_ind_label, fontsize=16).length(d.unit*.5)
    d += Resistor().down().label(sym_R_label, fontsize=16).length(d.unit*.5)
    d += Dot()
    d.pop()

    return(d)

def dibujar_tanque_serie(d, sym_ind_label='', sym_cap_label=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(inductor_lbl, sp.Number ):
        inductor_lbl = to_latex(inductor_lbl)
    else:
        inductor_lbl = str_to_latex(inductor_lbl)
    
    d.push()
    d += Dot()
    d += Inductor().right().label(inductor_lbl, fontsize=16)
    d.pop()
    d += Line().up().length(d.unit*.5)
    d += Resistor().right().label(sym_R_label, fontsize=16)
    d += Line().down().length(d.unit*.5)
    d += Dot()
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_tanque_RL_derivacion(d, sym_R_label='', inductor_lbl=''):
    
    if isinstance(sym_R_label, sp.Number ):
        sym_R_label = to_latex(sym_R_label)
    else:
        sym_R_label = str_to_latex(sym_R_label)
    
    if isinstance(inductor_lbl, sp.Number ):
        inductor_lbl = to_latex(inductor_lbl)
    else:
        inductor_lbl = str_to_latex(inductor_lbl)
    
    d.push()
    d += Dot()
    d += Inductor().down().label(inductor_lbl, fontsize=16).length(d.unit*.5)
    d += Resistor().down().label(sym_R_label, fontsize=16).length(d.unit*.5)
    d += Dot()
    d.pop()

    return(d)

def dibujar_tanque_serie(d, inductor_lbl='', capacitor_lbl=''):
    
    if isinstance(inductor_lbl, sp.Number ):
        inductor_lbl = to_latex(inductor_lbl)
    else:
        inductor_lbl = str_to_latex(inductor_lbl)
    
    if isinstance(capacitor_lbl, sp.Number ):
        capacitor_lbl = to_latex(capacitor_lbl)
    else:
        capacitor_lbl = str_to_latex(capacitor_lbl)
    
    d.push()
    d += Dot()
    d += Capacitor().right().label(capacitor_lbl, fontsize=16)
    d.pop()
    d += Line().up().length(d.unit*.5)
    d += Inductor().right().label(inductor_lbl, fontsize=16)
    d += Line().down().length(d.unit*.5)
    d += Dot()
    d.push()
    d += Gap().down().label( '' )
    d += Line().left()
    d.pop()

    return(d)

def dibujar_tanque_derivacion(d, inductor_lbl='', capacitor_lbl=''):
    
    if isinstance(inductor_lbl, sp.Number ):
        inductor_lbl = to_latex(inductor_lbl)
    else:
        inductor_lbl = str_to_latex(inductor_lbl)
    
    if isinstance(capacitor_lbl, sp.Number ):
        capacitor_lbl = to_latex(capacitor_lbl)
    else:
        capacitor_lbl = str_to_latex(capacitor_lbl)
    
    d.push()
    d += Dot()
    d += Capacitor().down().label(capacitor_lbl, fontsize=16).length(d.unit*.5)
    d += Inductor().down().label(inductor_lbl, fontsize=16).length(d.unit*.5)
    d += Dot()
    d.pop()

    return(d)


'''
    Bloque de funciones para la síntesis gráfica de imitancias
'''

def remover_polo_sigma( imm, sigma, isImpedance = True,  isRC = True,  sigma_zero = None ):
    '''
    Se removerá el residuo en sobre el eje $\sigma$ (sigma) de la impedancia (zz) 
    o admitancia (yy) de forma completa, o parcial en el caso que se especifique una 
    sigma_i.
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ Z_{R}= Z - \frac{k_i}{s + \sigma_i} $$
    
    siendo 

    $$ k=\lim\limits _{s\to -\sigma_i} Z (s + \sigma_i) $$
    
    En cuanto se especifique sigma_i, la remoción parcial estará definida 
    como

    $$ Z_{R}\biggr\rfloor_{s=-\sigma_i}= 0 = Z - \frac{k_i}{s + \sigma_i}\biggr\rfloor_{s=-\sigma_i} $$
    
    siendo 
    
    $$ k = Z.(\frac{)s + \sigma_i)\biggr\rfloor_{s=-\sigma_i} $$
    

    Parameters
    ----------
    zz o yy: Symbolic
        Impedancia o admitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un polo de orden 1 en \omega.
    omega_zero : Symbolic
        Frecuencia a la que la imitancia será cero luego de la remoción.

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k : Symbolic
        Valor del residuo.
    '''

    if isImpedance:
        zz = imm
    else:
        yy = imm

    if sigma_zero is None:
        # remoción total
        
        if isImpedance:
            if isRC:
                kk = sp.limit(zz*(s + sigma), s, -sigma)
            else:
                # RL
                kk = sp.limit(zz*(s + sigma)/s, s, -sigma)
                
        else:
            if isRC:
                kk = sp.limit(yy*(s + sigma)/s, s, -sigma)
            else:
                kk = sp.limit(yy*(s + sigma), s, -sigma)
        
        if kk.is_negative:
            assert('Residuo negativo. Verificar Z/Y RC/RL')
        
    else:
        # remoción parcial
        if isImpedance:
            if isRC:
                kk = sp.simplify(sp.expand(zz*(s + sigma))).subs(s, -sigma_zero)
            else:
                kk = sp.simplify(sp.expand(zz*(s + sigma)/s)).subs(s, -sigma_zero)
            
        else:
            if isRC:
                kk = sp.simplify(sp.expand(yy*(s + sigma)/s)).subs(s, -sigma_zero)
            else:
                kk = sp.simplify(sp.expand(yy*(s + sigma))).subs(s, -sigma_zero)

        if kk.is_negative:
            assert('Residuo negativo. Verificar Z/Y RC/RL')
    
    # extraigo kk
    if isImpedance:
        if isRC:
            # Z_RC        
            R = kk/sigma
            CoL = 1/kk
            kk  = kk/(s+sigma)
        else:
            # Z_RL        
            R = kk
            CoL = kk/sigma
            kk  = kk*s/(s+sigma)
        
    else:

        if isRC:
            # Y_RC        
            CoL = kk/sigma
            R = 1/kk
            kk  = kk*s/(s+sigma)
        else:
            # Y_RL
            R = sigma/kk
            CoL = 1/kk
            kk  = kk/(s+sigma)
        

    if isImpedance:
        imit_r = sp.factor(sp.simplify(sp.expand(zz - kk)))
    
    else:
    
        imit_r = sp.factor(sp.simplify(sp.expand(yy - kk)))

    return( [imit_r, kk, R, CoL] )

def remover_polo_jw( imit, omega = None , isImpedance = True, omega_zero = None ):
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

    if omega is None:
        # busco el primer polo finito en imit sobre el jw
        
        _, den = (imit).as_numer_denom()
        faux = sp.factor_list(den)
        
        if sp.degree(faux[1][0][0]) == 2:
            
            tt = faux[1][0][0].as_ordered_terms()
            
            # el último término sería omega**2. Cada factor sería
            # s**2 + omega**2
            omega = sp.sqrt(tt[-1])

    if omega_zero is None:
        # remoción total
        # kk = sp.limit(imit*(s**2+omega**2)/s, s**2, -omega**2)
        kk = sp.simplify(sp.expand(imit*(s**2+omega**2)/s)).subs(s**2, -(omega**2) )
        
    else:
        # remoción parcial
        kk = sp.simplify(sp.expand(imit*(s**2+omega**2)/s)).subs(s**2, -(omega_zero**2) )

    
    if isImpedance:
        # Z_LC
        L = kk/omega**2
        C = 1/kk
        
    else:
        # Y_LC
        C = kk/omega**2
        L = 1/kk

    kk = kk * s / (s**2+omega**2)
    
    # extraigo kk
    imit_r = sp.factor(sp.simplify(sp.expand(imit - kk)))

    return( [imit_r, kk, L, C] )

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
        k_cero = sp.limit(imit*s, s, 0)
        
    else:
        # remoción parcial
        k_cero = sp.simplify(sp.expand(imit*s)).subs(s**2, -(omega_zero**2) )

    k_cero = k_cero/s
    
    # extraigo C3
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_cero)))

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

    k_inf = k_inf * s

    # extraigo C3
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_inf)))

    return( [imit_r, k_inf] )

def remover_valor( imit, sigma_zero):
    '''
    Se removerá un valor constante de la imitancia ($I$) de forma 
    que al removerlo, la imitancia luego de la remoción ($I_R$) tenga 
    un cero en sigma_zero. Es decir:

    $$ I_{R}\biggr\rfloor_{s = -\sigma_z} = 0 = (I - k_{\infty}^{'})\biggr\rfloor_{s = -\sigma_z} $$
    
    siendo 
    
    $$ k_{\infty}^{'}= I\biggr\rfloor_{s = -\sigma_z} $$

    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un valor constante en infinito (mayor a su valor en s=0).
        
    omega_zero : Symbolic
        Frecuencia a la que la imitancia será cero luego de la remoción.

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    # remoción parcial
    k_prima = sp.simplify(sp.expand(imit)).subs(s, -sigma_zero)
    
    # extraigo k_prima
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_prima)))

    return( [imit_r, k_prima] )

def remover_valor_en_infinito( imit ):
    '''
    Se removerá un valor constante en infinito de la imitancia ($I$) de forma 
    completa. 
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ I_R = I - k_{\infty}  $$
    
    siendo 

    $$ k_{\infty}=\lim\limits _{s\to\infty}I $$

    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un valor constante en infinito (mayor a su valor en s=0).

    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    # remoción total
    k_inf = sp.limit(imit, s, sp.oo)

    # extraigo k_inf
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k_inf)))

    return( [imit_r, k_inf] )

def remover_valor_en_dc( imit ):
    '''
    Se removerá un valor constante en continua (s=0) de la imitancia ($I$) de forma 
    completa. 
    Como resultado de la remoción, quedará otra función racional definida
    como:
        
    $$ I_R = I - k_0  $$
    
    siendo 

    $$ k_0 = \lim\limits _{s \to 0}I $$
    
    Parameters
    ----------
    imit : Symbolic
        Imitancia que se utilizará para la remoción. Es una función racional 
        simbólica que tendrá un valor constante en infinito (mayor a su valor en s=0).
        
    Returns
    -------
    imit_r : Symbolic
        Imitancia luego de la remoción
    k_inf : Symbolic
        Valor del residuo en infinito
    '''

    # remoción total
    k0 = sp.limit(imit, s, 0)
        
    # extraigo k0
    imit_r = sp.factor(sp.simplify(sp.expand(imit - k0)))

    return( [imit_r, k0] )


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

    k = sp.poly(num,s).LC() / sp.poly(den,s).LC()
    
    roots_num = sp.roots(num)

    poly_acc = sp.Rational('1')
    
    for this_root in roots_num.keys():
        
        if sp.re(this_root) <= 0:
            
            # multiplicidad
            mult = roots_num[this_root]

            if mult > 1:
                poly_acc *= (s-this_root)**sp.Rational(mult/2)
            else:
                poly_acc *= (s-this_root)
                
            

    num = sp.simplify(sp.expand(poly_acc))

    roots_num = sp.roots(den)
    
    poly_acc = sp.Rational('1')

    for this_root in roots_num.keys():
        
        if sp.re(this_root) <= 0:
            
            # multiplicidad
            mult = roots_num[this_root]

            if mult > 1:
                poly_acc *= (s-this_root)**sp.Rational(mult/2)
            else:
                poly_acc *= (s-this_root)
    
    poly_acc = sp.simplify(sp.expand(poly_acc))

    return(sp.simplify(sp.expand(sp.sqrt(k) * num/poly_acc))) 


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


def pretty_print_lti(num, den = None, displaystr = True):
    
    if den is None:
        this_lti = num
    else:
        this_lti = TransferFunction(num, den)
    
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



def analyze_sys( all_sys, sys_name = None, img_ext = 'none', same_figs=True, annotations = True, digital = False, fs = 2*np.pi):
    
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
        fig_id, axes_hdl = bodePlot(all_sys[ii], fig_id, axes_hdl, filter_description = sys_name[ii], digital = digital, fs = fs)

    if img_ext != 'none':
        plt.savefig('_'.join(sys_name) + '_Bode.' + img_ext, format=img_ext)

    # fig_id = 6
    # axes_hdl = ()

    # for ii in range(cant_sys):
    #     fig_id, axes_hdl = bodePlot(all_sys[ii], fig_id, axes_hdl, filter_description = sys_name[ii])

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

            analog_fig_id, analog_axes_hdl = pzmap(thisFilter, filter_description=sys_name[ii], fig_id = analog_fig_id, axes_hdl=analog_axes_hdl, annotations = annotations, digital = digital, fs = fs)
            
        else:
                
            if all_sys[ii].dt is None:
                analog_fig_id, analog_axes_hdl = pzmap(all_sys[ii], filter_description=sys_name[ii], fig_id = analog_fig_id, axes_hdl=analog_axes_hdl, annotations = annotations)
            else:
                digital_fig_id, digital_axes_hdl = pzmap(all_sys[ii], filter_description=sys_name[ii], fig_id = digital_fig_id, axes_hdl=digital_axes_hdl, annotations = annotations)
            

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
        fig_id, axes_hdl = GroupDelay(all_sys[ii], fig_id, filter_description = sys_name[ii], digital = digital, fs = fs)
    
    # axes_hdl.legend(sys_name)

    axes_hdl.set_ylim(bottom=0)

    if img_ext != 'none':
        plt.savefig('_'.join(sys_name) + '_GroupDelay.'  + img_ext, format=img_ext)



def pzmap(myFilter, annotations = False, filter_description = None, fig_id='none', axes_hdl='none'):
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
    if filter_description is None:
        poles = plt.plot(p.real, p.imag, 'x', markersize=9)
    else:
        poles = plt.plot(p.real, p.imag, 'x', markersize=9, label=filter_description)
    
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

    if not(filter_description is None):
       axes_hdl.legend()

    return fig_id, axes_hdl
    

def GroupDelay(myFilter, fig_id='none', filter_description=None, npoints = 1000, digital = False, fs = 2*np.pi):

    w_nyq = 2*np.pi*fs/2
    
    if isinstance(myFilter, np.ndarray):
        # SOS section
        cant_sos = myFilter.shape[0]
        phase = np.empty((npoints, cant_sos+1))
        sos_label = []
        
        for ii in range(cant_sos):
            
            num, den = one_sos2tf(myFilter[ii,:])
            thisFilter = TransferFunction(num, den)
            
            if digital:
                w, _, phase[:,ii] = thisFilter.bode(np.linspace(10**-2, w_nyq, npoints))
            else:
                w, _, phase[:,ii] = thisFilter.bode(np.logspace(-2,2,npoints))
            
            sos_label += [filter_description + ' - SOS {:d}'.format(ii)]
        
        # whole filter
        thisFilter = sos2tf_analog(myFilter)
        
        if digital:
            w, _, phase[:,cant_sos] = thisFilter.bode(np.linspace(10**-2, w_nyq, npoints))
        else:
            w, _, phase[:,cant_sos] = thisFilter.bode(np.logspace(-2,2,npoints))
        
        sos_label += [filter_description]
        
        filter_description = sos_label
        
    else:
        # LTI object
        cant_sos = 0
        
        if myFilter.dt is None:
            w, _, phase = myFilter.bode(np.logspace(-2,2,npoints))
        else:
            w, _, phase = myFilter.bode(np.linspace(10**-2, w_nyq, npoints))
        
        if isinstance(filter_description, str):
            filter_description = [filter_description]


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

    if digital:
        aux_hdl = plt.plot(w[1:] / w_nyq , groupDelay, label=filter_description)    # Bode phase plot
    else:
        aux_hdl = plt.semilogx(w[1:], groupDelay, label=filter_description)    # Bode phase plot

    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)
    
    
    if digital:

        plt.gca().set_xlim([0, 1])
        
        plt.xlabel('Frecuencia normalizada a Nyq [#]')
    else:
        plt.xlabel('Angular frequency [rad/sec]')
    
    plt.ylabel('Group Delay [sec]')
    plt.title('Group delay')

    axes_hdl = plt.gca()
    
    if not(filter_description is None):
        # axes_hdl.legend( filter_description )
        axes_hdl.legend()

    return fig_id, axes_hdl

def bodePlot(myFilter, fig_id='none', axes_hdl='none', filter_description=None, npoints = 1000, digital = False, fs = 2*np.pi ):

    w_nyq = 2*np.pi*fs/2
    
    if isinstance(myFilter, np.ndarray):
        # SOS section
        cant_sos = myFilter.shape[0]
        mag = np.empty((npoints, cant_sos+1))
        phase = np.empty_like(mag)
        sos_label = []
        
        for ii in range(cant_sos):
            
            num, den = one_sos2tf(myFilter[ii,:])
            thisFilter = TransferFunction(num, den)
            if digital:
                w, mag[:, ii], phase[:,ii] = thisFilter.bode(np.linspace(10**-2, w_nyq,npoints))
            else:
                w, mag[:, ii], phase[:,ii] = thisFilter.bode(np.logspace(-2,2,npoints))
                
            sos_label += [filter_description + ' - SOS {:d}'.format(ii)]
        
        # whole filter
        thisFilter = sos2tf_analog(myFilter)

        if digital:
            w, mag[:, cant_sos], phase[:,cant_sos] = thisFilter.bode(np.linspace(10**-2, w_nyq, npoints))
        else:
            w, mag[:, cant_sos], phase[:,cant_sos] = thisFilter.bode(np.logspace(-2,2,npoints))
            
        sos_label += [filter_description]
        
        filter_description = sos_label
        
    else:
        # LTI object
        cant_sos = 0
        
        if myFilter.dt is None:
            # filtro analógico normalizado
            w, mag, phase = myFilter.bode(np.logspace(-2,2,npoints))
        else:
            w, mag, phase = myFilter.bode(np.linspace(10**-2, w_nyq, npoints))
        
        if isinstance(filter_description, str):
            filter_description = [filter_description]
        

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

    if digital:
        if filter_description is None:
            aux_hdl = plt.plot(w / w_nyq, mag)    # Bode magnitude plot
        else:
            aux_hdl = plt.plot(w / w_nyq, mag, label=filter_description)    # Bode magnitude plot
    else:
        if filter_description is None:
            aux_hdl = plt.semilogx(w, mag)    # Bode magnitude plot
        else:
            aux_hdl = plt.semilogx(w, mag, label=filter_description)    # Bode magnitude plot
    
    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)
#    plt.xlabel('Angular frequency [rad/sec]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Magnitude response')
    
    if not(filter_description is None):
        # mag_ax_hdl.legend( filter_description )
        mag_ax_hdl.legend()

        
    plt.sca(phase_ax_hdl)
    
    if digital:
        if filter_description is None:
            aux_hdl = plt.plot(w / w_nyq, np.pi/180*phase)    # Bode phase plot
        else:
            aux_hdl = plt.plot(w / w_nyq, np.pi/180*phase, label=filter_description)    # Bode phase plot
            
    else:
        if filter_description is None:
            aux_hdl = plt.semilogx(w, np.pi/180*phase)    # Bode phase plot
        else:
            aux_hdl = plt.semilogx(w, np.pi/180*phase, label=filter_description)    # Bode phase plot
    
    
    # Scale axes to fit
    ylim = plt.gca().get_ylim()

    # presentar la fase como fracciones de \pi
    ticks = np.linspace(start=np.round(ylim[0]/np.pi)*np.pi, stop=np.round(ylim[1]/np.pi)*np.pi, num = 5, endpoint=True)

    ylabs = []
    for aa in ticks:
        
        if aa == 0:
            ylabs += ['0'] 
        else:
            bb = Fraction(aa/np.pi).limit_denominator(1000000)
            if np.abs(bb.numerator) != 1:
                if np.abs(bb.denominator) != 1:
                    str_aux = r'$\frac{{{:d}}}{{{:d}}} \pi$'.format(bb.numerator, bb.denominator)
                else:
                    str_aux = r'${:d}\pi$'.format(bb.numerator)
                    
            else:
                if np.abs(bb.denominator) == 1:
                    if np.sign(bb.numerator) == -1:
                        str_aux = r'$-\pi$'
                    else:
                        str_aux = r'$\pi$'
                else:
                    if np.sign(bb.numerator) == -1:
                        str_aux = r'$-\frac{{\pi}}{{{:d}}}$'.format(bb.denominator)
                    else:
                        str_aux = r'$\frac{{\pi}}{{{:d}}}$'.format(bb.denominator)
                    
            ylabs += [ str_aux ]
            
    plt.yticks(ticks, labels = ylabs )
    
    if cant_sos > 0:
        # distinguish SOS from total response
        [ aa.set_linestyle(':') for aa in  aux_hdl[:-1]]
        aux_hdl[-1].set_linewidth(2)
    
    plt.grid(True)

    if digital:

        plt.gca().set_xlim([0, 1])
        
        plt.xlabel('Frecuencia normalizada a Nyq [#]')
    else:
        plt.xlabel('Angular frequency [rad/sec]')
    plt.ylabel('Phase [rad]')
    plt.title('Phase response')
    
    if not(filter_description is None):
        # phase_ax_hdl.legend( filter_description )
        phase_ax_hdl.legend()
    
    return fig_id, axes_hdl
    

def plot_plantilla(filter_type = 'lowpass', fpass = 0.25, ripple = 0.5, fstop = 0.6, attenuation = 40, fs = 2 ):
    
    # para sobreimprimir la plantilla de diseño de un filtro
    
    xmin, xmax, ymin, ymax = plt.axis()
    
    # banda de paso digital
    plt.fill([xmin, xmin, fs/2, fs/2],   [ymin, ymax, ymax, ymin], 'g', alpha= 0.2, lw=1, label = 'bw digital') # pass
    
    if filter_type == 'lowpass':
    
        fstop_start = fstop
        fstop_end = xmax
        
        fpass_start = xmin
        fpass_end   = fpass
    
        plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
        plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass
    
    elif filter_type == 'highpass':
    
        fstop_start = xmin
        fstop_end = fstop 
        
        fpass_start = fpass
        fpass_end   = xmax
    
        plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
        plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass
    
    
    elif filter_type == 'bandpass':
    
        fstop_start = xmin
        fstop_end = fstop[0]
        
        fpass_start = fpass[0]
        fpass_end   = fpass[1]
        
        fstop2_start = fstop[1]
        fstop2_end =  xmax
        
        plt.fill( [fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # stop
        plt.fill( [fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass
        plt.fill( [fstop2_start, fstop2_end,   fstop2_end, fstop2_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k') # stop
        
    elif filter_type == 'bandstop':
    
        fpass_start = xmin
        fpass_end   = fpass[0]
    
        fstop_start = fstop[0]
        fstop_end = fstop[1]
        
        fpass2_start = fpass[1]
        fpass2_end   = xmax
            
        plt.fill([fpass_start, fpass_start, fpass_end, fpass_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k', label = 'plantilla') # pass
        plt.fill([fstop_start, fstop_end,   fstop_end, fstop_start], [-attenuation, -attenuation, ymax, ymax], '0.9', lw=1, ls = '--', ec = 'k') # stop
        plt.fill([fpass2_start, fpass2_start, fpass2_end, fpass2_end],   [ymin, -ripple, -ripple, ymin], '0.9', lw=1, ls = '--', ec = 'k') # pass
    
    
    plt.axis([xmin, xmax, np.max([ymin, -100]), np.max([ymax, 5])])
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    
    plt.show()
    
    


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
        
        # bode in dB
        maxima_tf[si] = 10**(np.max(mag)/20)
    
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

