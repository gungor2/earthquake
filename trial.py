# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:26:03 2019

@author: gungor2
"""

import matplotlib.pyplot as plt
import numpy as np
# the equation is 2x + 4 = y --> 2x + 4 - y = 0 --> w1 = 2, w2 = -1 + b = 4
x = list(range(-10,10))
x = np.asarray(x) 
y = 2 * x + 4

yy = 2 * x + 2


w =[2,1]
y1 = list(range(-16,21))

min_y = min(y)
max_y = max(y)

min_x = min(x)
max_x = max(x)


ax_x = x
ax_y = np.repeat(0,len(x))

ay_y = list(range(min_y,max_y))
ay_x = np.repeat(0,len(ay_y))


fig = plt.figure(figsize=(10,10))


    
fig_name = 'try.png'
plt.plot(x,y,'*',x,yy,ax_x,ax_y,ay_x,ay_y)
plt.savefig(fig_name)
plt.grid()
plt.show()
plt.close()

