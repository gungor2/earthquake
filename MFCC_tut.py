# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:47:45 2019

@author: gungor2
"""

from python_speech_features import mfcc
from python_speech_features import logfbank

rate = 4*10**6
sig = seg.values

mfcc_feat = mfcc(sig,rate)

fbank_feat = logfbank(sig,rate)
