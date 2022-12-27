import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def Hz2Barks(f):
    bark = 13 * np.arctan(0.00076 * f*44100) + 3.5 * np.arctan( np.power(f*44100 / 7500, 2)) # na to alla3oume
    return bark

def make_mp3_analysisfb(h, M):
    L = h.shape[0]
    H = np.ndarray([L, M])
    for rowIdx in range(L):
        for colIdx in range(M):
            H[rowIdx][colIdx] = h[rowIdx]*np.cos(((2*colIdx + 1)*np.pi*rowIdx)/(2*M) + ((2*colIdx + 1)*np.pi)/4)
    return H

def get_column_fourier(H):
    Hf = np.ndarray(H.shape, dtype = np.cdouble)
    for i in range(Hf.shape[1]):
        Hf[:, i] = sp.fft.fft(H[:, i])
    return Hf

def plot_in_hz_in_db_units(Hf):
    plt.yscale('log')
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0 : 255]]
        imagPart = [x.imag for x in Hf[:, i][0 : 255]]
        plt.plot([x**2 + y**2 for x, y in zip(realPart, imagPart)])
    plt.show()

def plot_in_barks_in_db_units(Hf):
    plt.yscale('log')
    for i in range(Hf.shape[1]):
        realPartinHz = [x.real for x in Hf[:, i][0 : 255]]
        imagPartinHz = [x.imag for x in Hf[:, i][0 : 255]]
        realPartinBarks = []
        imagPartinBarks = []
        barksAxes = []
        for f in range(255):
            fhz = f/512
            barks = Hz2Barks(fhz)
            barksAxes.append(barks)
            realPartinBarks.append(realPartinHz[int(barks*512)])
            print(int(barks * 512))
            imagPartinBarks.append(imagPartinHz[int(barks*512)])
        if(i < 3):
            plt.plot(barksAxes, [np.sqrt(x**2 + y**2) for x, y in zip(realPartinBarks, imagPartinBarks)])
    plt.show()

data_h = np.load('h.npy', allow_pickle=True)
h = data_h[()]['h']
H = make_mp3_analysisfb(h, 32)
Hf = get_column_fourier(H)
#plot_in_hz_in_db_units(Hf)
plot_in_barks_in_db_units(Hf)