import numpy as np
from scipy.linalg import solve, cholesky, solve_triangular, cho_factor
from scipy.signal import freqz
from scipy.integrate import trapezoid  # Reemplazo para np.trapz
import matplotlib.pyplot as plt
import warnings
from numbers import Integral, Real


# Función para graficar la convergencia
def plot_convergence(debug_info, R=5):
    """Grafica la convergencia del algoritmo incluyendo respuestas en frecuencia
    
    Parámetros:
    -----------
    debug_info : dict
        Información de depuración devuelta por l1_filter_design
    R : int, opcional
        Número de respuestas en frecuencia a mostrar (por defecto 5)
    """
    history = debug_info['history']
    total_iterations = len(history)
    
    # Determinar los índices de las iteraciones a mostrar
    if total_iterations <= R:
        show_indices = range(total_iterations)
    else:
        step = max(1, total_iterations // R)
        show_indices = list(range(0, total_iterations-1, step)) + [total_iterations-1]
    
    # Configurar la figura
    plt.figure(figsize=(15, 12))
    
    # Gráfico 1: Evolución del error L1
    plt.subplot(3, 2, 1)
    l1_errors = [entry['l1_error'] for entry in history]
    plt.plot(l1_errors, 'o-')
    # Resaltar las iteraciones que mostraremos
    for i in show_indices:
        plt.plot(i, l1_errors[i], 'ro', markersize=8)
    plt.title('Evolución del Error L1')
    plt.xlabel('Iteración')
    plt.ylabel('Error L1')
    plt.grid(True)
    
    # Gráfico 2: Criterio de parada
    plt.subplot(3, 2, 2)
    stop_criteria = [entry['stop_criterion'] for entry in history]
    plt.semilogy(stop_criteria, 'o-')
    for i in show_indices:
        plt.plot(i, stop_criteria[i], 'ro', markersize=8)
    plt.title('Criterio de Parada (log)')
    plt.xlabel('Iteración')
    plt.ylabel('|d^T g|')
    plt.grid(True)
    
    # Gráfico 3: Tamaño de paso
    plt.subplot(3, 2, 3)
    step_sizes = [entry['step_size'] for entry in history]
    plt.plot(step_sizes, 'o-')
    for i in show_indices:
        plt.plot(i, step_sizes[i], 'ro', markersize=8)
    plt.title('Tamaño de Paso (Armijo)')
    plt.xlabel('Iteración')
    plt.ylabel('Gamma')
    plt.grid(True)
    
    # Gráfico 4: Número de ceros
    plt.subplot(3, 2, 4)
    num_zeros = [entry['num_zeros'] for entry in history]
    plt.plot(num_zeros, 'o-')
    for i in show_indices:
        plt.plot(i, num_zeros[i], 'ro', markersize=8)
    plt.title('Número de Ceros en Bandas')
    plt.xlabel('Iteración')
    plt.ylabel('Número de ceros')
    plt.grid(True)
    
    # Gráfico 5: Respuestas en frecuencia seleccionadas
    plt.subplot(3, 2, (5,6))
    w = np.linspace(0, np.pi, 1024)
    
    # Obtener las frecuencias de corte del diseño
    band_edges = debug_info.get('band_edges', None)
    
    cant_bases = debug_info.get('cant_bases', None)
    
    # Graficar cada respuesta seleccionada
    for idx, i in enumerate(show_indices):
        entry = history[i]
        a = entry['coefficients']
        M = len(a) - 1
        h = np.zeros(2*M + 1)
        h[cant_bases] = a[0]
        for n in range(1, cant_bases+1):
            h[cant_bases - n] = h[cant_bases + n] = a[n] / 2
        
        _, H = freqz(h, worN=w)
        color = plt.cm.viridis(idx / len(show_indices))
        label = f'Iter {i} (L1={entry["l1_error"]:.2e})'
        plt.plot(w/np.pi, 20*np.log10(np.abs(H)), color=color, label=label)
    
    if band_edges is not None:
        [plt.axvline(be, color='red', linestyle='--', alpha=0.5) for be in band_edges[1:-1]]
        
        # plt.axvline(wp/np.pi, color='red', linestyle='--', alpha=0.5)
        # plt.axvline(ws/np.pi, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Respuesta en Frecuencia en Iteraciones Clave')
    plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def case_iv(V, D, cant_bases, delta1=1e-12, delta2=1e12):
    """Implementación exacta del Caso IV con Cholesky modificado según [39]"""
    t = V.shape[1]  # Número de ceros (t < cant_bases+1)
    Hk = V @ (D @ V.T)  # Hk = V·D·Vᵀ (size (cant_bases+1)x(cant_bases+1), rango t
    
    # 1. Calcular autovalores para determinar la corrección necesaria
    eigvals = np.linalg.eigvalsh(Hk)
    min_eig = np.min(eigvals)
    
    # 2. Aplicar corrección solo si es necesario (Gill-Murray-Wright)
    if min_eig < delta1:
        # Calcular la matriz de corrección diagonal C
        C = np.zeros(cant_bases+1)
        tau = max(delta1, 1e-8)  # Threshold de positividad
        
        # Factorización LDLᵀ con correcciones
        L = np.eye(cant_bases+1)
        for j in range(cant_bases+1):
            # Perturbar la diagonal si es necesario
            if Hk[j,j] < tau:
                C[j] = tau - Hk[j,j]
                Hk[j,j] = tau
            
            L[j,j] = np.sqrt(Hk[j,j])
            for i in range(j+1, cant_bases+1):
                L[i,j] = Hk[i,j] / L[j,j]
                for k in range(j+1, i+1):
                    Hk[i,k] -= L[i,j] * L[k,j]
        
        Hk = L @ L.T  # Reconstruir la matriz definida positiva
        return Hk, f"Case IV (Modified Cholesky, t={t}, C_max={np.max(C):.1e})"
    else:
        return Hk, f"Case IV (Already PD, t={t})"

def safe_solve(Hk, g, eps=1e-12):
    """Resuelve Hk*d = -g con regularización automática"""
    try:
        # Intentar resolver sin regularización primero
        
        with warnings.catch_warnings():
            # Suprimir específicamente LinAlgWarning
            warnings.filterwarnings('ignore', category=np.linalg.LinAlgWarning)

            d = solve(Hk, -g, assume_a='pos')
            
        if np.any(np.isnan(d)):
            raise np.linalg.LinAlgError
        return d
    except np.linalg.LinAlgError:
        # Regularización basada en la norma de Hk
        reg = eps * np.linalg.norm(Hk, np.inf)
        return solve(Hk + reg * np.eye(Hk.shape[0]), -g, assume_a='pos')
        
def svd_solve(Hk, g, tol=1e-12):
    U, s, Vh = np.linalg.svd(Hk)
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    return -Vh.T @ (s_inv * (U.T @ g))
    
def solve_direction(Hk, g, method='adaptive'):
    if method == 'adaptive':
        return safe_solve(Hk, g)  # Usar la opción 2
    elif method == 'svd':
        return svd_solve(Hk, g)   # Usar la opción 3 para máxima precisión

def find_zero_crossing_indices(w_eval: np.ndarray, zeros: np.ndarray) -> np.ndarray:
    """
    Encuentra los índices en w_eval más cercanos a las frecuencias de los ceros.
    
    Parameters:
        w_eval (np.ndarray): Vector de frecuencias donde evaluar (debe estar ordenado)
        zeros (np.ndarray): Frecuencias de los cruces por cero
        
    Returns:
        np.ndarray: Índices de w_eval más cercanos a cada cero
    """
    # Asegurar que w_eval esté ordenado (requisito para searchsorted)
    if not np.all(np.diff(w_eval) >= 0):
        raise ValueError("w_eval debe estar ordenado ascendentemente")
    
    # Encontrar índices de inserción (donde cada cero debería estar en w_eval)
    idx = np.searchsorted(w_eval, zeros)
    
    # Ajustar índices en los bordes
    idx = np.clip(idx, 1, len(w_eval)-1)
    
    # Comparar distancia con el punto anterior y siguiente
    left_diff = np.abs(zeros - w_eval[idx-1])
    right_diff = np.abs(w_eval[idx] - zeros)
    z_idx = idx - (left_diff < right_diff).astype(int)
    
    return z_idx

def fir_design_L1(order, band_edges, desired, weight = None, grid_density = 16, 
                  fs = 2.0, filter_type = 'multiband', end_tol=1e-6, max_iter = 100, debug=False):
    """
    Algoritmo de Parks-McClellan para el diseño de filtros FIR de fase lineal
    utilizando un criterio minimax. El algoritmo está basado en RERMEZ_FIR de 
    :ref:`Tapio Saramaki y Lars Whannamar <DSPMatlab20>` y el detallado trabajo
    en el material suplementario de :ref:`Thomas Holton <holton21>`. La imple_
    mentación del algoritmo ha sido ampliamente modificada con fines didácticos
    respecto a la version original de Saramaki y Parks McClellan.
    
    Parameters
    -----------
    order : TransferFunction
        Orden del filtro a diseñar. El tamaño del filtro será de *orden+1*.
    band_edges : array_like
        Los límites de cada banda indicada en la plantilla de diseño del filtro.
        Habrá dos valores, principio y fin, por cada banda definida en *fr_desiredired*.
        Ej: [0., 0.3, 0.7, 1.] Para un pasabajos con corte en 0.3
    desired : array_like
        El valor numérico deseado por cada banda. Ej: [1.0, 0.] para un pasabajos.
    weight : array_like
        Un valor postivo que pesará cada banda al momento de calcular el error.
    grid_density : int, numeric
        Un entero que indicará por cuanto interpolar la respuesta del filtro al
        calcular el error del filtro. El valor de interpolación se calcula 
        *aproximadamente* por grid_density*orden/2. Por defecto se usa 16.
    fs : float, numeric
        Frecuencia de muestreo a la que se implementará el filtro digital. Por
        defecto se usa 2.0, es decir se normaliza a la f. de Nyquist.
    filter_type : string, 
        Un string que identifica el filtro que se diseñará. Se admiten tres 
        posibilidafr_desired: 'multiband' o 'm'. Filtros FIR tipo 1 o 2 de propósitos 
        generales. 'differentiator' o 'd', se utilizará para diseñar filtro FIR 
        derivadores de tipo 3 o 4 dependiendo el orden. Finalmente, 'hilbert' o
        'h' para implementar filtros FIR que permiten calcular la parte 
        imaginaria de una señal analítica. Es decir tener una transferencia 
        aproximadamente constante y una rotación constante de pi/2 para todas 
        las frecuencias.
    end_tol : float, opcional
        Tolerancia para el criterio de parada
    max_iter : int, numeric
        Cantidad máxima de iteraciones del algoritmo de Remez para hallar las 
        frecuencias extremas.
    debug : boolean
        Un valor booleano para activar la depuración de la propia función.

    Returns
    --------
    h_coeffs : array_like
        Los coeficientes de la respuesta al impulso del filtro FIR diseñado.
    err : float, numeric
        Error máximo obtenido de la iteración del algoritmo Remez.
    w_extremas : array_like
        Las frecuencias extremas obtenidas de la iteración del algoritmo Remez.

    Raises
    ------
    ValueError
        Si no se cumple con el formato y valores indicados en la documentación.

    See Also
    -----------
    :func:``
    :func:``

    Examples
    --------
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 

    Notes:
    -------
    .. _Grossmann07:

    L. D. Grossmann and Y. C. Eldar, "An $L_1$-Method for the Design of Linear-Phase FIR Digital Filters," in IEEE Transactions on Signal Processing, vol. 55, no. 11, pp. 5253-5266, Nov. 2007, doi: 10.1109/TSP.2007.896088.
    	
    """

    if not (isinstance(order, (Integral, Real)) and order > 0 ):
        raise ValueError("El argumento 'order' debe ser un número positivo.")
    
    order = int(order)
              
    if not (isinstance(grid_density, (Integral, Real)) and grid_density > 0 ):
        raise ValueError("El argumento 'grid_density' debe ser un número positivo.")

    if not (isinstance(fs, Real) and fs > 0 ):
        raise ValueError("El argumento 'fs' debe ser un número positivo.")

    if not (isinstance(max_iter, (Integral, Real)) and max_iter > 0 ):
        raise ValueError("El argumento 'max_iter' debe ser un número positivo.")

    if not isinstance(debug, bool):
        raise ValueError('debug debe ser un booleano')


    valid_filters = ['multiband', 'lowpass', 'highpass', 'bandpass', 
                     'bandstop', 'notch',
                     'h', 'd', 'm', 'lp', 'hp', 'lp', 'bp',
                     'differentiator', 'hilbert']

    if not isinstance(filter_type, str):
        raise ValueError("El argumento 'filter_type' debe ser un string de %s" % (valid_filters))

	#==========================================================================
	#  Find out jtype that was used in the PM code.
	#  This not necessary but simplifies the undertanding of this code snippet.
	#==========================================================================
    if filter_type.lower().startswith('d'):
        jtype = 2  # Differentiator
    elif 'hilbert' == filter_type.lower() or 'h' == filter_type.lower():
        jtype = 3  # Hilbert transformer
    else: 
        jtype = 1  # Multiband filter

	#==========================================================================
	# Determine the filter cases and cant_bases, the number of basis functions to be 
	# used in the Remez algorithm 
	# In the below, filtercase=1,2,3,4 is used for making it easier to 
	# understand this code snippet.   
	#==========================================================================
    # Determine the filter cases and cant_bases
    if jtype == 1:
        if order % 2 == 0:
            filtercase = 1  # Even order and even symmetry multiband filter
        else:
            filtercase = 2  # Odd order and even symmetry multiband filter 
    else:
        if order % 2 == 0:
            # Even order and odd symmetry -> a Hilbert transforer or a 
            # differentiator (jtype indicates)
            filtercase = 3  
        else:
            # Odd order and odd symmetry -> a Hilbert transforer or a 
            # differentiator (jtype indicates)
            filtercase = 4  

    if filter_type not in valid_filters:
        raise ValueError('El tipo de filtro debe ser uno de %s, no %s'
                         % (valid_filters, filter_type))

    if not isinstance(band_edges, (list, np.ndarray)):
        raise ValueError("El argumento 'band_edges' debe ser una lista o un array de numpy.")

    if not isinstance(desired, (list, np.ndarray)):
        raise ValueError("El argumento 'fr_desiredired' debe ser una lista o un array de numpy.")
    
    if not isinstance(weight, (type(None), list, np.ndarray)):
        raise ValueError("El argumento 'weight' debe ser una lista o un array de numpy.")

    # Chequear si la plantilla de requerimientos del filtro está bien armada.
    ndesired = len(desired)
    nedges = len(band_edges)
    nbands = nedges // 2

    if isinstance(weight ,type(None)):
        weight = np.ones(nbands)
        
    if isinstance(weight, list):
        weight = np.array(weight)

    nweights = len(weight)

    if ndesired != nedges:
        raise ValueError(f"Debe haber tantos elementos en 'fr_desired' {ndesired} como en band_edges:{nedges}")

    if jtype == 1:
        # multibanda 
        if nweights != nbands:
            raise ValueError(f"Debe haber tantos elementos en 'weight' {nweights} como cantidad de bandas {nbands}")

    if jtype == 2 or jtype == 3:
        # derivador y hilbert
        if nbands != 1:
            raise ValueError(f"Debe haber en una sola banda definida para FIR tipo {filter_type}, hay {nbands} bandas")

    # normalizar respecto a Nyquist
    band_edges = np.array(band_edges) / (fs/2)

    desired = np.array(desired)

            
	# cant_bases - number of basis functions 
    cant_coeffs = order + 1
    
    if filtercase == 1 or filtercase == 3:
        M = (cant_coeffs-1) // 2
        # cantidad de frecuencias extremas.
        # cant_bases = M + 1
        cant_bases = M
        
    if filtercase == 2 or filtercase == 4:
        M = cant_coeffs // 2
        # cantidad de frecuencias extremas.
        cant_bases = M 



    # Determine fr_grid, fr_desired, and fr_weight
    total_freq_bins = grid_density * cant_bases
    freq_resolution = 1.0 / total_freq_bins
    # full resolution (fr) fr_grid, desired and wieight arrays
    fr_grid = []
    fr_desired = []
    fr_weight = []
    # indexes of the band-edges corresponding to the fr freq. fr_grid array
    band_edges_idx = []

    min_number_fr_grid = int(np.min( (cant_bases * .1, 20) ))

    for ll in range(nbands):
        
        number_fr_grid = np.max( (min_number_fr_grid, int(np.ceil((band_edges[2 * ll + 1] - band_edges[2 * ll]) / freq_resolution))) )
        
        fr_grid_more = np.linspace(band_edges[2 * ll], band_edges[2 * ll + 1], number_fr_grid + 1)
        
        # Adjust fr_grid for harmful frequencies at omega = 0 
        if ll == 0 and (filtercase == 3 or filtercase == 4) and fr_grid_more[0] < freq_resolution:
            fr_grid_more = fr_grid_more[1:]
            number_fr_grid -= 1

        # Adjust fr_grid for harmful frequencies at omega = 1
        if ll == nbands - 1 and (filtercase == 2 or filtercase == 3) and fr_grid_more[-1] > 1 - freq_resolution:
            fr_grid_more = fr_grid_more[:-1]
            number_fr_grid -= 1

        #
        band_edges_idx.extend([len(fr_grid)])
        fr_grid.extend(fr_grid_more)
        band_edges_idx.extend([len(fr_grid)-1])

        if jtype == 2:
            # differentiator
            
            des_more = desired[2*ll+1] * fr_grid_more * np.pi
            
            if np.abs(desired[2*ll]) < 1.0e-3:
                wt_more = weight[ll] * np.ones(number_fr_grid + 1)
            else:
                wt_more = weight[ll] / (fr_grid_more * np.pi)
        else:
            # others

            wt_more = weight[ll] * np.ones(number_fr_grid + 1)
            if desired[2 * ll + 1] != desired[2 * ll]:
                des_more = np.linspace(desired[2 * ll], desired[2 * ll + 1], number_fr_grid + 1)
            else:
                des_more = desired[2 * ll] * np.ones(number_fr_grid + 1)

        fr_desired.extend(des_more)
        fr_weight.extend(wt_more)

    fr_grid = np.array(fr_grid)
    fr_desired = np.array(fr_desired)
    fr_weight = np.array(fr_weight)
    band_edges_idx = np.array(band_edges_idx)

	#==========================================================================
	# Modify fr_desired and fr_weight depending on the filter case
	#========================================================================== 
    # Este es un elegante truco para hacer una sola función de optimización
    # de Remez para todos los tipos de FIRs. 
    # Ver :ref:`Thomas Holton supplimentary material <holton21>`.
    # 
    
    if filtercase == 2:
        fr_desired /= np.cos(np.pi * fr_grid / 2)
        fr_weight *= np.cos(np.pi * fr_grid / 2)
    if filtercase == 4:
        fr_desired /= np.sin(np.pi * fr_grid / 2)
        fr_weight *= np.sin(np.pi * fr_grid / 2)
    if filtercase == 3:
        fr_desired /= np.sin(np.pi * fr_grid)
        fr_weight *= np.sin(np.pi * fr_grid)

    # Parámetros del algoritmo
    beta = 0.7  # Más suave
    sigma = 1e-3  # Menos estricto
    delta1 = 1e-12
    delta2 = 1e12
    mu = 1e-6
    
    # Preparar información de depuración
    if debug:
        debug_info = {
            'history': [],
            'cant_bases': cant_bases,
            'band_edges': band_edges * np.pi
        }
    
    # Inicialización (Paso 1)
    z_init = np.pi * (2 * np.arange(1, cant_bases+2) - 1) / (2*(cant_bases+1))  # Corregido
    
    D_init = np.zeros_like(z_init)
    for ll in range(nbands):

        bAux = np.bitwise_and( z_init >= band_edges[ll*2]* np.pi , z_init <= band_edges[ll*2+1]* np.pi )
        D_init[bAux] = desired[2*ll]  # Respuesta ideal
        
    
    # Matriz de cosenos para interpolación
    V = np.array([[np.cos(n*z) for n in range(cant_bases+1)] for z in z_init])
    a = solve(V, D_init)  # Coeficientes iniciales
    
    
    # Frecuencias para evaluación
    # w = fr_grid_more * np.pi
    # w_eval = w[(w <= wprad) | (w >= wsrad)]
    w_eval = fr_grid * np.pi
    
    for iteration in range(max_iter):
        
        # Paso 2: Encontrar ceros de la función de error (método mejorado)
        Aw = np.zeros_like(w_eval)
        for ii, aa in enumerate(a):
            Aw += aa * np.cos( ii * w_eval )
        E = Aw - fr_desired
        sign_E = np.sign(E)
        
        
        # Detección robusta de ceros
        zero_crossings = np.where(np.diff(np.sign(E)))[0]
        zeros = []
        for idx in zero_crossings:
            if idx+1 >= len(w_eval):
                continue
            x1, x2 = w_eval[idx], w_eval[idx+1]
            y1, y2 = E[idx], E[idx+1]
            
            if y1 != y2 :  # Evitar división por cero
                zero = x1 - y1*(x2-x1)/(y2-y1)
                # if np.any( w_eval == zero ):  # Solo en bandas de interés
                zeros.append(zero)
        
        zeros = np.array(zeros)
        t = len(zeros)
        
        z_idx = find_zero_crossing_indices(w_eval, zeros)
        
        # if t < cant_bases+1:
        #     # Añadir puntos estratégicos además de los ceros encontrados
        #     extra_points = np.linspace(0, wprad, M//3)
        #     extra_points = np.concatenate([extra_points, np.linspace(wsrad, np.pi, M//3)])
        #     zeros = np.unique(np.sort(np.concatenate([zeros, extra_points])))
        #     zeros = zeros[:cant_bases+1]  # Tomar exactamente cant_bases+1 puntos
        #     t = len(zeros)


        if debug and iteration % max_iter//4 == 0:
            plt.figure()
            plt.plot(w_eval/np.pi, E, label='Error')
            plt.plot(zeros/np.pi, np.zeros_like(zeros), 'ro', label='Ceros detectados')
            [plt.axvline(be, color='green', linestyle='--') for be in band_edges[1:-1]]
            # plt.axvline(ws, color='green', linestyle='--')
            plt.title(f'Iteración {iteration}: Ceros de la función de error')
            plt.legend()
            plt.show()
        
        # Construir D y V según el paper
        if t == 0:
            Hk = np.eye(cant_bases+1)
            hk_type =  "Identidad (sin ceros)"

        else:
            
            # Calcular D
            E_primes = -np.sum([n * a[n] * np.sin(n * zeros) for n in range(1, cant_bases+1)], axis=0)
            D_vals = 2 * fr_weight[z_idx] / (np.abs(E_primes) + 1e-12)
            D = np.diag(D_vals)
            
            # Verificar condiciones de la Tabla I
            d_min, d_max = np.min(D_vals), np.max(D_vals)
            cos_zeros = np.cos(zeros)
            cos_diff = np.abs(np.subtract.outer(cos_zeros, cos_zeros) + np.eye(t))
            min_cos_diff = np.min(cos_diff)
            
            # Caso III: Hessian exacta
            if (t == M + 1) and (delta1 <= d_min) and (d_max <= delta2) and (min_cos_diff > mu):
                V = np.array([[np.cos(n * z) for z in zeros] for n in range(cant_bases+1)])
                Hk = V.T @ D @ V
                hk_type =  "Hessian exacto(Case III)"
            
            # Caso IV: Cholesky modificado
            elif (0 < t < M + 1) or (delta1 <= d_min and d_max <= delta2) or (min_cos_diff <= mu):
                
                V = np.array([[np.cos(n * z) for z in zeros] for n in range(cant_bases+1)])
                Hk, hk_type = case_iv(V, D, cant_bases)
            
            # Casos I y II: Usar matriz identidad
            else:
                return np.eye(cant_bases+1),
                Hk = np.eye(cant_bases+1)
                hk_type =  f"Identidad (Case I/II: d_min={d_min:.1e}, d_max={d_max:.1e}, cos_diff={min_cos_diff:.1e})"
        
        # Paso 3: Calcular gradiente (versión más precisa)
        
        # Aw = np.zeros_like(w_eval)
        # for ii, aa in enumerate(a):
        #     Aw += aa * np.cos( ii * w_eval )
        # E = Aw - np.where(w_eval <= wprad, 1, 0)
        # sign_E = np.sign(E)
        
        g = np.zeros(cant_bases+1)
        for n in range(cant_bases+1):
            integrand = fr_weight * np.cos(n*w_eval) * sign_E
            g[n] = trapezoid(integrand, w_eval)
       
        # Paso 4: Verificar criterio de parada
        try:
            d = solve_direction(Hk, g, method='adaptive')
        except:
            d = -g  # Retroceder a dirección de descenso si hay problemas
        
        stop_criterion = np.abs(d @ g)
        
        if debug:
            l1_error = trapezoid(fr_weight * np.abs(E), w_eval)
            debug_info['history'].append({
                'iteration': iteration,
                'coefficients': a.copy(),
                'zeros': zeros.copy(),
                'gradient': g.copy(),
                'Hk_type': hk_type,
                'stop_criterion': stop_criterion,
                'l1_error': l1_error,
                'step_size': None,
                'num_zeros': t
            })
            
        # Paso 5: Búsqueda de línea mejorada
        gamma = 1.0
        armijo_iter = 0
        l1_old = trapezoid(fr_weight * np.abs(E), w_eval)
        sufficient_decrease = False
        
        while armijo_iter < 20:  # Límite de iteraciones

            a_new = a + gamma * d
            
            Aw_new = np.zeros_like(w_eval)
            for ii, aa in enumerate(a_new):
                Aw_new += aa * np.cos( ii * w_eval )
            E_new = Aw_new - fr_desired
            
            l1_new = trapezoid(fr_weight * np.abs(E_new), w_eval)
            
            if (l1_new - l1_old) <= (sigma * gamma * (d @ g)):
                sufficient_decrease = True
                break
                
            gamma *= beta
            armijo_iter += 1
        
        if not sufficient_decrease:
            # Si Armijo falla, intentar dirección de gradiente
            gamma = 1e-4  # Paso pequeño
            d = -g  # Dirección de gradiente
            a_new = a + gamma * d
        
        if debug:
            debug_info['history'][-1]['step_size'] = gamma
            debug_info['history'][-1]['armijo_iterations'] = armijo_iter
            
        # Paso 6: Actualización
        a = a_new

        # if stop_criterion < end_tol or (debug and iteration > 10 and rel_change < 1e-6):
        if stop_criterion < end_tol:
            if debug:
                print(f"Convergencia alcanzada en iteración {iteration}")
            break
    
    # Convertir coeficientes a a la respuesta al impulso h
    
    #======================================================
    # Construir el filtro a partir de los coeficientes "a"
	#======================================================
    a_coeffs = a
    
    cant_acoeffs = len(a_coeffs)

    # convertir los coeficientes según el tipo de FIR
    if filtercase == 1:
        
        a_coeffs [1:] = a_coeffs[1:]/2
        h_coeffs = np.concatenate((a_coeffs[::-1], a_coeffs[1:]))
    
    if filtercase == 2:
        
        last_coeff = cant_acoeffs
        cant_hcoeff = 2*cant_acoeffs
        h_coeffs = np.zeros(cant_hcoeff)
        h_coeffs[cant_hcoeff-1] = a_coeffs[last_coeff-1]/4
        h_coeffs[last_coeff] = a_coeffs[0] /2 + a_coeffs[1]/4
        h_coeffs[last_coeff+1:cant_hcoeff-1]= (a_coeffs[1:last_coeff-1] + a_coeffs[2:last_coeff])/4
            
        h_coeffs[:last_coeff] = h_coeffs[last_coeff:][::-1]

        
    if filtercase == 3:
        
        cant_hcoeff = 2*cant_acoeffs+1
        h_coeffs = np.zeros(cant_hcoeff)
        last_coeff = cant_acoeffs # punto de simetría, demora del filtro


        h_coeffs[0:2] = a_coeffs[last_coeff-2:][::-1]/4
        h_coeffs[2:last_coeff-1] = ((a_coeffs[1:last_coeff-2] - a_coeffs[3:last_coeff])/4)[::-1]
        h_coeffs[last_coeff-1] = a_coeffs[0]/2 - a_coeffs[2]/4
        
        h_coeffs[last_coeff+1:] = (-1.)*h_coeffs[:last_coeff][::-1]

    if filtercase == 4:
        
        last_coeff = cant_acoeffs
        cant_hcoeff = 2*cant_acoeffs
        h_coeffs = np.zeros(2*cant_acoeffs)
        h_coeffs[cant_hcoeff-1] = a_coeffs[last_coeff-1]/4
        h_coeffs[last_coeff] = a_coeffs[0]/2 - a_coeffs[1]/4
        h_coeffs[last_coeff+1:cant_hcoeff-1]= (a_coeffs[1:last_coeff-1] - a_coeffs[2:last_coeff])/4
            
        h_coeffs[:last_coeff] = -1. * h_coeffs[last_coeff:][::-1]
    
    if debug:
        debug_info['final_coeffs'] = a
        debug_info['final_zeros'] = zeros
        return h_coeffs, debug_info
    else:
        return h_coeffs
    

# Ejemplo de uso con la nueva implementación
if __name__ == "__main__":
    # Diseñar un filtro pasa-bajos de orden 64 con depuración
    N = 64
    wp1 = 0.4
    wp2 = 0.6
    ws1 = 0.35
    ws2 = 0.65

    plt.close('all')
    
    # niter = 400
    # ftype = 'multiband'
    # band_e = [0, wp2, ws2, 1.]
    # desired = [ 1., 1., 0., 0.]
    
    # band_e = [0, ws1, wp1, 1.]
    # desired = [ 0., 0., 1., 1.]

    # band_e = [0, ws1, wp1, wp2, ws2, 1.]
    # desired = [ 0., 0., 1., 1., 0., 0.]

    # ftype = 'd'
    # band_e = [0, 1.]
    # desired = [ 0., 1.]


    N = 64
    niter = 100
    ftype = 'h'
    band_e = [0.05, 0.95]  # Bandas de diseño (en términos de frecuencia normalizada)
    desired = [1., 1.]  # Respuesta ideal del derivador (pendiente lineal en la banda de paso)

    # Usar la nueva implementación mejorada
    h, debug_info = fir_design_L1(N, band_edges = band_e, desired = desired, 
                                  filter_type = ftype, 
                                  max_iter=niter, debug=True)
    
    # Graficar información de convergencia
    plot_convergence(debug_info, R=5)
    
    # Mostrar información de la última iteración
    if debug_info and 'history' in debug_info and debug_info['history']:
        last_iter = debug_info['history'][-1]
        print("\nInformación de la última iteración:")
        print(f"Iteración: {last_iter['iteration']}")
        print(f"Error L1: {last_iter['l1_error']:.6e}")
        print(f"Criterio de parada: {last_iter['stop_criterion']:.6e}")
        print(f"Tamaño de paso: {last_iter['step_size']:.6e}")
        print(f"Número de ceros: {last_iter['num_zeros']}")
        print(f"Tipo de matriz Hk: {last_iter['Hk_type']}")
    
    # Graficar la respuesta final
    w, H = freqz(h, worN=8192)
    plt.figure(figsize=(10, 5))
    plt.plot(w/np.pi, 20*np.log10(np.abs(H)))
    [plt.axvline(be, color='red', linestyle='--') for be in band_e[1:-1]]

    # plt.axvline(wp, color='red', linestyle='--', label='Banda de paso')
    # plt.axvline(ws, color='red', linestyle='--', label='Banda de rechazo')
    plt.title('Respuesta Final del Filtro L₁ Mejorado')
    plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

