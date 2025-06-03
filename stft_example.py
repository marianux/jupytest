import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, welch

def generar_senal_stft(f1=50, f2=400, center1=0.25, center2=0.75,
                       width1=0.1, width2=0.1, fs=1000., T=1.0):
    
    # # Parámetros
    # f1, f2 = 50, 200
    # center1, center2 = 0.5, 1.5
    # width1, width2 = 0.1, 0.1
    # fs, T = 1000, 1
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    
    # Señal
    def gaussian_window(center, width, t):
        return np.exp(-((t - center) ** 2) / (2 * width ** 2))
    
    sig1 = np.sin(2 * np.pi * f1 * t) * gaussian_window(center1, width1, t)
    sig2 = np.sin(2 * np.pi * f2 * t) * gaussian_window(center2, width2, t)
    signal = sig1 + sig2
    
    # Estimación por Welch
    f_welch, Pxx = welch(signal, fs=fs, nperseg=256)
    
    # STFT
    f, t_stft, Zxx = stft(signal, fs=fs, nperseg=256)
    
    # Crear figura y ejes
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Señal
    ax1.plot(t, signal)
    ax1.set_title(f"Señal compuesta: {f1} Hz @ {center1}s y {f2} Hz @ {center2}s")
    ax1.set_ylabel("Amplitud")
    ax1.set_xlim(t[0], t[-1])
    
    # Subplot 2: Welch
    ax2.semilogy(f_welch, Pxx)
    ax2.set_title("Estimación espectral por Welch")
    ax2.set_ylabel("PSD [V²/Hz]")
    ax2.set_xlabel("Frecuencia [Hz]")
    
    # Espectrograma
    pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
    ax3.set_title("STFT (Espectrograma)")
    ax3.set_ylabel("Frecuencia [Hz]")
    ax3.set_xlabel("Tiempo [s]")
    
    # Colorbar en eje externo
    cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
    fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
    plt.show()


generar_senal_stft(f1=100., f2=400., center1=0.25, center2=0.75, width1=0.05, width2=0.05)
generar_senal_stft(f1=100., f2=400., center1=0.75, center2=0.25, width1=0.05, width2=0.05)

generar_senal_stft(f1=100., f2=400., center1=0.25, center2=0.75, width1=0.005, width2=0.01)
generar_senal_stft(f1=100., f2=400., center1=0.75, center2=0.25, width1=0.005, width2=0.01)
