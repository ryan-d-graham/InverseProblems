import pywt
import soundfile as sf
from os import system
import numpy as np #Numerical Python stuffs
from scipy.signal import correlate
from python_speech_features import mfcc, logfbank
import pandas as pd #Pandas Data stuffs
from pandas_ods_reader import read_ods #Convert Libre Office Spreadsheet data to Pandas DataFrame object
import matplotlib.pyplot as plt

#filename = input("Enter filename (file_name.wav): ")
filename = 'test1.wav'
amps, meta = sf.read(filename)
cut = 0
amps = amps[cut:]
amps = amps[5000:20000] #keystroke pulses

mfcc_feat = mfcc(amps,meta)
fbank_feat = logfbank(amps,meta)
plt.imshow(mfcc_feat)
plt.show()
plt.imshow(fbank_feat)
plt.show()

jump = 200 #do what sf.blocks does with the raw amps data
varList = [np.var(amps[i:i+jump]) for i in range(len(amps)-jump)]
plt.plot(varList)
plt.show()

varFft = np.fft.fft(varList)
varSpec = np.real(varFft*np.conj(varFft))[4:68]

plt.plot(varSpec)
plt.show()



