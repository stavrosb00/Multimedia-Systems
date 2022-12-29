import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from frame import *
from nothing import *

def Hz2Barks(f):
    bark = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan( np.power(f / 7500, 2))
    return bark

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
        Hf[:, i] = sp.fft.fft(H[:, i])
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

def codec0(wavin, h, M, N):
    subwavinsTotal = wavin.shape[0]//(M*N)
    Ytot = np.ndarray([N * subwavinsTotal, M])
    H = make_mp3_analysisfb(h, M)

    for i in range(subwavinsTotal):
        subwav = wavin[i*M*N:(i+1)*(M*N)]
        subwav = np.append([0 for _ in range(511)], subwav) # zero padding for linear convolution
        Y = frame_sub_analysis(subwav, H, N)
        Yc = donothing(Y)
        Ytot[i*N:(i+1)*N, :] = Yc

    #######################################
    G = make_mp3_synthesisfb(h, M)
    buffSize = M * N
    totalSize = Ytot.shape[0] * Ytot.shape[1]
    xhat = np.ndarray([totalSize])
    for i in range(Ytot.shape[0]//N):
        Yc = Ytot[i*N:(i+1)*N, :]
        Yh = idonothing(Yc)
        xhat[i*buffSize : (i+1)*buffSize] = frame_sub_synthesis(Yh, G)

    return xhat

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
xhat = codec0(wavin, h, M, N)
xhatscaled = np.int16(xhat / np.max(np.abs(xhat)) * 32767)
write("testDec1.wav", fs, xhatscaled)
