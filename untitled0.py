# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:56:23 2019

@author: gungor2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
np.random.seed(444)

data_train = pd.read_csv(".\\LANL-Earthquake-Prediction\\train.csv")

#data_train.describe()
#
#data_train.head()
#
#plt.plot(data_train.iloc[:,0])
#
#plt.plot(data_train.iloc[:,1])
#
#plt.plot(data_train.iloc[:,0],data_train.iloc[:,1],'*')



nrows = data_train.shape[0]
ini = data_train.iloc[0,1]
quakes = []
for i in range(1,nrows):
    if data_train.iloc[i,1]>ini:
        ini = data_train.iloc[i,1]
        quakes.append(i-1)
        print(i)
    
#plt.plot(data_train.iloc[4500000:4750000,0],data_train.iloc[4500000:4750000,1])
