import numpy as np
from scipy.signal import convolve, freqz
import matplotlib.pyplot as plt

def generate_q_filters_scipy(j_max):
    """
    Genera los filtros Q^j(ω) usando scipy.signal.
    
    Args:
        j_max (int): Máximo valor de j (j > 1)
        
    Returns:
        list: Lista de tuplas (j, coefficients) donde coefficients es un array numpy
    """
    if j_max <= 1:
        raise ValueError("j debe ser mayor que 1")
    
    filters = []
    
    # Coeficientes base
    # Filtro H(ω) de la ecuación (A2)
    h_coeff = np.array([1/8, 3/8, 3/8, 1/8])  # índices: -1,0,1,2
    
    # Filtro G(ω) = 2(e^{iω} - 1)
    g_coeff = np.array([-2, 2])  # índices: 0,1
    
    # Caso j=1 (Q^1(ω) = G(ω))
    q1 = g_coeff
    filters.append((1, q1))
    
    # Caso j=2 (dado explícitamente en el paper)
    q2 = np.array([-1/4, -3/4, -1/2, 1/2, 3/4, 1/4])  # índices: -1,0,1,2,3,4
    filters.append((2, q2))
    
    # Casos j > 2
    for j in range(3, j_max + 1):
        # Calcular P(ω) = H(2^{j-2}ω) * ... * H(ω)
        p = np.array([1.0])  # Inicializar con delta[n]
        
        for k in range(j-1):
            # Escalar H por 2^k (insertar ceros)
            h_scaled = np.zeros( len(h_coeff) * 2**k )
            h_scaled[::2**k] = h_coeff
            
            # Convolución acumulativa
            p = convolve(p, h_scaled, mode='full')
        
        # # Normalizar por 8^{j-1}
        # p = p / (8**(j-1))
        
        # Calcular G(2^{j-1}ω) (insertar ceros)
        g_scaled = np.zeros(2**(j-1) + 1 )
        g_scaled[0] = -2
        g_scaled[-1] = 2
        
        # Calcular Q^j(ω) = G(2^{j-1}ω) * P(ω)
        qj = convolve(g_scaled, p, mode='full')
        
        # Eliminar ceros insignificantes
        qj = np.trim_zeros(qj, 'b')
        qj = np.trim_zeros(qj, 'f')
        
        filters.append((j, qj))
    
    return filters

def plot_frequency_response_scipy(filters):
    """
    Grafica la respuesta en frecuencia usando scipy.signal.freqz.
    
    Args:
        filters (list): Lista de filtros devuelta por generate_q_filters_scipy()
    """
    plt.figure(figsize=(12, 8))
    plt.title("Respuesta en Frecuencia de los Filtros Q^j(ω)")
    plt.xlabel("Frecuencia normalizada (×π rad/muestra)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for j, coeffs in filters:
        # Calcular respuesta en frecuencia
        w, h = freqz(coeffs, worN=2048)
        
        # Convertir a dB
        magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        # magnitude_db = np.abs(h)
        
        # Graficar
        plt.plot(w/np.pi, magnitude_db, label=f"Q^{j}(ω)")
    
    plt.xlim(0, 1)
    plt.ylim(-60, 15)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    
    # Generar filtros
    j_max = 6
    q_filters = generate_q_filters_scipy(j_max)
    
    # Mostrar coeficientes
    for j, coeffs in q_filters:
        print(f"\nFiltro Q^{j}(ω):")
        print("Longitud:", len(coeffs))
        print("Coeficientes:")
        print(coeffs)
    
    # Graficar respuesta en frecuencia
    plot_frequency_response_scipy(q_filters)
    
    