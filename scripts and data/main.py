import numpy as np
import scipy.fft as sp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from frame import *
from nothing import *

# utility fun
def signaltonoise(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def Hz2Barks(f):
    bark = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan( np.power(f / 7500, 2))
    return bark

#LEVEL 3.1 FUNCTIONS 
def make_mp3_analysisfb(h, M):
    L = h.shape[0]
    H = np.ndarray([L, M])
    for rowIdx in range(L):
        for colIdx in range(M):
            H[rowIdx][colIdx] = h[rowIdx]*np.cos(((2*colIdx + 1)*np.pi*rowIdx)/(2*M) + ((2*colIdx + 1)*np.pi)/4)
    return H

def make_mp3_synthesisfb(h, M):
    H = make_mp3_analysisfb(h, M)
    G = H
    for colIdx in range(M):
        for rowIdx in range(H.shape[0]):
            G[rowIdx, colIdx] = H[rowIdx, colIdx] * (H.shape[0] - 1 - rowIdx)

    return G

def get_column_fourier(H):
    Hf = np.ndarray(H.shape, dtype = np.cdouble)
    for i in range(Hf.shape[1]):
        Hf[:, i] = sp.fft(H[:, i])
    return Hf

def plot_in_hz_in_db_units(Hf):
    fs = 44100
    fstep = fs/512
    frequencyXaxis = [x*fstep for x in range(0, 255)]
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0 : 255]]
        imagPart = [x.imag for x in Hf[:, i][0 : 255]]

        plt.plot(frequencyXaxis, [10*np.log10(x ** 2 + y ** 2) for x, y in zip(realPart, imagPart)])
    plt.show()

def plot_in_barks_in_db_units(Hf):
    fs = 44100
    fstep = fs / 512
    frequencyXaxis = [x * fstep for x in range(0, 255)]
    barksXaxis = [Hz2Barks(x) for x in frequencyXaxis]
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0: 255]]
        imagPart = [x.imag for x in Hf[:, i][0: 255]]

        plt.plot(barksXaxis, [10*np.log10(x ** 2 + y ** 2) for x, y in zip(realPart, imagPart)])
    plt.show()

def coder0(wavin, h, M, N):
    subwavinsTotal = wavin.shape[0]//(M*N)
    Ytot = np.ndarray([N * subwavinsTotal, M])
    H = make_mp3_analysisfb(h, M)

    wavin = np.append(wavin, [0 for _ in range(512)])  # padded

    for i in range(subwavinsTotal):
        subwav = wavin[i*(M*N):i*M*N + M*(N-1)+512]
        Y = frame_sub_analysis(subwav, H, N)
        Yc = donothing(Y)
        Ytot[i*N:(i+1)*N, :] = Yc
    
    return Ytot

def decoder0(Ytot, h, M, N):
    G = make_mp3_synthesisfb(h, M)
    buffSize = M * N
    totalSize = Ytot.shape[0] * Ytot.shape[1]
    xhat = np.ndarray([totalSize])
    for i in range(Ytot.shape[0]//N):
        Yc = Ytot[i*N:(i+1)*N + h.shape[0] // M, :]
        Yh = idonothing(Yc)
        xhat[i*buffSize : (i+1)*buffSize] = frame_sub_synthesis(Yh, G)

    return xhat

def codec0(wavin, h, M, N):
    #4 early steps
    Ytot = coder0(wavin, h, M, N)
    #2 last steps
    xhat = decoder0(Ytot, h, M, N)
    
    return xhat, Ytot
##################
# LEVEL 3.2 FUNCTIONS DCT-IV
# https://docs.scipy.org/doc/scipy/tutorial/fft.html#type-iv-dct

#LEVEL 3.1 FILTERBANK EXECUTION
# 1-3
fs = 44100
M = 32
N = 36
data_h = np.load('h.npy', allow_pickle=True)
h = data_h[()]['h']
H = make_mp3_analysisfb(h, M)
Hf = get_column_fourier(H)
#plot_in_hz_in_db_units(Hf)
#plot_in_barks_in_db_units(Hf)

# 4
wavin = read("myfile.wav")
wavin = np.array(wavin[1],dtype=float)

xhat, Ytot = codec0(wavin, h, M, N)
xhatscaled = np.int16(xhat * 32767 / np.max(np.abs(xhat)))
write("testDec2.wav", fs, xhatscaled)

# error projection

fig1 = plt.figure(1)
ax1 = fig1.gca()
plt.subplot(1, 2, 1)
plt.plot(wavin[0:2000])
plt.title("MyFile Wavin")
plt.subplot(1, 2, 2)
plt.plot(xhatscaled[0: 2000])
plt.title("Decoded Wavin")
plt.show()

#error = wavin - xhat
#e_snr = signaltonoise(error)
# print(e_snr)

#fig2 = plt.figure(2)
#ax2 = fig2.gca()
#plt.title("Error between input and decoded wavin file(SNR = %1.7f)" %e_snr)
#plt.plot(error)
#plt.show()




#LEVEL 3.2 DCT IV EXECUTION
