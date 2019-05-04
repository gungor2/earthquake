# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:28:24 2019

@author: gungor2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
from scipy import signal


np.random.seed(444)

train = pd.read_csv('./LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

rows = 150000
segments = int(np.floor(train.shape[0] / rows))

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)


y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

fs = 4*10**6
images = []
for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = pd.Series(seg['acoustic_data'].values)
    f, t, Sxx = signal.spectrogram(x,fs)
    b = Sxx.reshape((129,669,1))
    images.append(b)

    

        
        
    y = np.mean(seg['time_to_failure'].values)
    
    y_tr.loc[segment, 'time_to_failure'] = y
    