# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:44:24 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:56:23 2019

@author: gungor2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from numpy import linalg
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






quakes = [5656573,50085877,104677355,138772452,187641819,218652629,245829584,307838916,338276286,375377847,419368879,
461811622,495800224,528777114,585568143,621985672]

slopes = []
for i in range(len(quakes)):

    
    if i==0:
        quake = data_train.iloc[0:quakes[0],:]

        
    else:
        quake = data_train.iloc[quakes[i-1]+1:quakes[i],:]


    data_x = list(range(0,quake.shape[0]))
    data_y = [1]*quake.shape[0]
    
    data_mat = np.column_stack((data_x,data_y))
    
    inv_ = linalg.pinv(data_mat)
    
    
    pre =  quake.iloc[:,1].values
    
    res =  inv_ @ pre
    print(res[0])
    slopes.append(res[0])
