# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:31:22 2019

@author: gungor2
"""

from scipy import signal
import matplotlib.pyplot as plt

fs = 4*10**6
f, t, Sxx = signal.spectrogram(seg.values.reshape(-1),fs)

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()