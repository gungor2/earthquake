# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:06:03 2019

@author: gungor2
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob

np.random.seed(444)


pathh = "./LANL-Earthquake-Prediction/test"
all_files = glob.glob(os.path.join(pathh,"*.csv"))


for i in range(len(all_files)):
    temp = pd.read_csv(all_files[i])
    fig = plt.figure(figsize=(10,10))
    fig_name = str(i) + '_test_acus_fig.png'
    plt.plot(list(range(len(temp.iloc[:,0]))),temp.iloc[:,0],'*')
    plt.title(all_files[len(all_files[i])-12:])
    plt.savefig(fig_name)
    plt.close()
    if i==0:
        all_df = temp
    else:
        fig = plt.figure(figsize=(10,10))
        all_df = all_df.append(temp)
        fig_name = str(i) + '_agg_test_acus_fig.png'
        plt.plot(list(range(len(all_df.iloc[:,0]))),all_df.iloc[:,0],'*')
        plt.title(all_files[len(all_files[i])-12:])
        plt.savefig(fig_name)
        plt.close()
        
    print(i)

