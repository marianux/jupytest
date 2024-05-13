import numpy as np

def LSE_FIR(Norder, Be, D):
    # Modification of Burrus first algorithm
    N = Norder + 1
    Be = Be / np.pi
    h = fir3(N, Be, D)
    return h

def fir3(N, f, m):
    """
    h = fir3(N, f, m) calculates an optimal least square error
    multiband FIR filter from a simple lowpass design.
    f is a vector of break frequencies and m is a vector
    of ideal frequency response values at those frequencies.
    The format is similar to that of "remez". An example of a
    bandpass filter:
    h = fir3(31, [0, 0.3, 0.33, 0.5, 0.6, 1], [0, 0, 1, 1, 0, 0])

    f and m must be the same even length. m must have pairs of
    equal values (the ideal filter must be a pice-wise constant).
    Remember that an even length filter must be zero at f = 1.
    The multiband filter is constructed from lowpass filters
    designed by fir3lp.m
    """
    L = len(f)
    if m[-1] == 0:
        h = np.zeros(N)  # Frequency response is zero at pi
    else:
        h = np.concatenate([np.zeros((N - 1) // 2), [m[-1]], np.zeros((N - 1) // 2)])

    while L > 2:
        h += (m[L - 2] - m[L - 1]) * fir3lp(N, f[L - 2], f[L - 1])  # Construct
        L -= 2

    return h

def fir3lp(N, f1, f2, p=None):
    """
    b = fir3lp(N, f1, f2, p) designs a linear phase lowpass FIR filter
    b(n) of length N with a least integral squared error approximation
    to an ideal lowpass filter with a passband from 0 to f1 and a stopband
    from f2 to 1. (in normalized Hertz) and a p-order spline transition
    band. If p is not given, a near optimal value is calculated as
    p = 0.62*N*d.
    """
    if p is None:
        p = 0.31 * N * (f2 - f1)  # Optimal spline power p

    w0 = np.pi * (f2 + f1) / 2  # Average band edge
    dw = np.pi * (f2 - f1) / 2  # Half transition width

    if N % 2 == 0:
        nx = np.arange(1 / 2, (N - 1) / 2 + 1)  # Even length index vector
    else:
        nx = np.arange(1, (N - 1) / 2 + 1)  # Odd length index vector

    h = (np.sin(w0 * nx)) / (np.pi * nx)  # LP filter with no transition

    if dw != 0 and p != 0:
        wd = (dw / p) * nx  # Weigthing function: wt
        wt = (np.sin(wd) / wd) ** p
        h *= wt

    if N % 2 == 0:
        b = np.concatenate([h[::-1], h])  # Even length output
    else:
        b = np.concatenate([h[::-1], [w0 / np.pi], h])  # Odd length output

    return b


import scipy.signal as sig
import matplotlib.pyplot as plt

wT = np.linspace(0,np.pi,1000)
Be = np.array([0, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.85, 0.9, 1])*np.pi
D = np.array([0, 0, 0.7, 0.7, 0.5, 0.5, 0, 0, 1, 1,])

# Must be even
Norder = 100;

h = LSE_FIR(Norder, Be, D);

H = sig.freqz(h, 1, wT);


plt.plot(wT, 20*np.log10(np.abs(H)+1e-12), label='FIR-LS {:d}'.format(H.shape[0]) )

# plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.title('FIR diseñado por métodos directos')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
# plt.axis([0, 500, -60, 5 ]);

# plt.grid()

# axes_hdl = plt.gca()
# axes_hdl.legend()
            
plt.figure(2)

phase_h = np.angle(H)

plt.plot(wT, phase_h, label='FIR-LS {:d}'.format(H.shape[0]))    # Bode phase plot

plt.title('FIR diseñado Fase')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Fase [rad]')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()




