# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:33:35 2019

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
np.random.seed(444)

data_train = pd.read_csv(".\\LANL-Earthquake-Prediction\\train.csv")



#nrows = data_train.shape[0]
#quakes = []
#for i in range(nrows-1):
#    if data_train.iloc[i,1]<data_train.iloc[i+1,1]:
#        quakes.append(i+1)
#        print(i)

    
#plt.plot(data_train.iloc[4500000:4750000,0],data_train.iloc[4500000:4750000,1])


quakes = [5656573,50085877,104677355,138772452,187641819,218652629,245829584,307838916,338276286,375377847,419368879,
461811622,495800224,528777114,585568143,621985672]

fig = plt.figure(figsize=(10,10))

for i in range(len(quakes)):
    
    print(i)
    if i==0:
        quake = data_train.iloc[0:quakes[0],:]

        
    else:
        quake = data_train.iloc[quakes[i-1]:quakes[i],:]
        
    
    fig_name = str(i) + '_acus_fig.png'
    plt.plot(quake.iloc[:,0],'*')
    plt.savefig(fig_name)
    plt.close()
    
    fig_name = str(i) + '_times_fig.png'
    plt.plot(quake.iloc[:,1],'*')
    plt.savefig(fig_name)
    plt.close()
    
    fig_name = str(i) + '_all_fig.png'
    plt.plot(quake.iloc[:,0],quake.iloc[:,1],'*')
    plt.savefig(fig_name)
    plt.close()
        


