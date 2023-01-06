import numpy as np
import scipy.fft as sp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from frame import *
from nothing import *

def Hz2Barks(f):
    """
    Converting Hertz to Barks 
    Params:
        f: Hertz
    Returns:
        bark
    """
    bark = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan( np.power(f / 7500, 2))
    return bark

#LEVEL 3.1 FILTERBANK FUNCTIONS 
def make_mp3_analysisfb(h, M):
    """
    Analyse the filterbank to M bands
    Params:
        h: Base function with L length of low pass filter
        M: bands to split
    Returns:
        H matrix LxM analysis filters per band(M collumns)
    """
    L = h.shape[0]
    H = np.ndarray([L, M])
    for rowIdx in range(L):
        for colIdx in range(M):
            H[rowIdx][colIdx] = h[rowIdx]*np.cos(((2*colIdx + 1)*np.pi*rowIdx)/(2*M) + ((2*colIdx + 1)*np.pi)/4)
    return H

def make_mp3_synthesisfb(h, M):
    """
    Compute the synthesis functions of filterbank for M bands
    Params:
        h: Base function with L length of low pass filter
        M: bands to split
    Returns:
        G matrix LxM synthesis filters per band(col)
    """
    H = make_mp3_analysisfb(h, M)
    G = H
    for colIdx in range(M):
        for rowIdx in range(H.shape[0]):
            G[rowIdx, colIdx] = H[rowIdx, colIdx] * (H.shape[0] - 1 - rowIdx)

    return G

def get_column_fourier(H):
    """
    Compute the Fast Fourier transformation of H matrix per each band
    Params:
        H: matrix LxM analysis filters per band(M collumns)
    Returns:
        Hf Fourier transformation of H matrix
    """
    Hf = np.ndarray(H.shape, dtype = np.cdouble)
    for i in range(Hf.shape[1]):
        Hf[:, i] = sp.fft(H[:, i])
    return Hf

def plot_in_hz_in_db_units(Hf):
    """
    Plotting M filters of Hf matrix based on Frequency x-axis
    Params:
        Hf: Fourier transformation of H matrix
    Returns:
        -
    """
    fs = 44100
    fstep = fs/512
    frequencyXaxis = [x*fstep for x in range(0, 255)]
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0 : 255]]
        imagPart = [x.imag for x in Hf[:, i][0 : 255]]

        plt.plot(frequencyXaxis, [10*np.log10(x ** 2 + y ** 2) for x, y in zip(realPart, imagPart)])
    plt.show()

def plot_in_barks_in_db_units(Hf):
    """
    Plotting M filters of Hf matrix based on Barks x-axis
    Params:
        Hf: Fourier transformation of H matrix
    Returns:
        -
    """
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
        if i == 100:
            print(Y)
            c = frameDCT(Y)
            #print(len(c))
            tempY = iframeDCT(c)
            #print(tempY.shape)
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
    #print(Ytot.shape)
    #2 last steps
    xhat = decoder0(Ytot, h, M, N)
    
    return xhat, Ytot
##################
# LEVEL 3.2 FUNCTIONS DCT-IV
# https://docs.scipy.org/doc/scipy/tutorial/fft.html#type-iv-dct
def frameDCT(Y):
    #36x32
    tempC = np.ndarray(Y.shape)
    for i in range(Y.shape[1]):
        tempC[:, i] = sp.dct(Y[:, i], type = 4)
    c = tempC.flatten('F')
    #print(c)
    return c

def iframeDCT(c):
    M = 32
    N = 36
    tempC = np.reshape(c, (N, M), 'F')
    Yh = np.ndarray((N, M))
    for i in range(M):
        Yh[:, i] = sp.idct(tempC[:, i], type = 4)
    #print(Yh)
    return Yh

# LEVEL 3.3 PSYCHOACOUSTIC MODEL 

def DCTpower(c):
    #sxesh 10
    P = 10 * np.log10(np.power(c, 2))
    return P

def Dksparse(Kmax):

    return D

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
xhatscaled = np.int16(xhat *  32767 / np.max(np.abs(xhat)))
#xhatscaled = xhatscaled * max(wavin)/max(xhatscaled)
write("testDec2.wav", fs, xhatscaled)

xhatscaled = read("testDec2.wav")
xhatscaled = np.array(xhatscaled[1],dtype=float)
xhatscaled = xhatscaled * np.max(np.abs(wavin)) / np.max(np.abs(xhatscaled))
print(xhatscaled.shape)
print(wavin.shape)

allRhos = np.ndarray((1000,1))

for i in range(1000):
    shifted_xhat = np.roll(xhatscaled, i)
    rho = np.corrcoef(wavin, shifted_xhat)
    #print(rho)
    allRhos[i] = rho[0][1]

    #rho[0][1]

maxRhoIdx = np.argmax(allRhos)
print(maxRhoIdx, allRhos[maxRhoIdx])
shifted_xhat = np.roll(xhatscaled, maxRhoIdx )
error = wavin - shifted_xhat
#error = wavin - xhatscaled

powS = np.mean(np.power(shifted_xhat, 2))
powN = np.mean(np.power(error, 2))
print(powS, powN)
snr = 10*np.log10((powS-powN)/powN)
# error projection

fig1 = plt.figure(1)
ax1 = fig1.gca()
plt.subplot(2, 1, 1)
plt.plot(wavin[4000:4100])
plt.title("MyFile Wavin")
plt.subplot(2, 1, 2)
plt.plot(shifted_xhat[4000:4100])
plt.title("Decoded Shifted Wavin")
plt.show()

fig2 = plt.figure(2)
ax2 = fig2.gca()
plt.title("Error between input and decoded wavin file(SNR = %1.5f dB)" %snr)
plt.plot(error[4000:4100])
plt.show()

#LEVEL 3.2 DCT IV EXECUTION
