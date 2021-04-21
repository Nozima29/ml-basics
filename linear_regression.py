# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:14:03 2021

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([1, 2, 3, 4])
y = np.array([1.5, 2.5, 3.5, 4])
n = len(x)
epoch = 5
X = np.empty(shape=(1, 4), dtype=float)
Y = np.empty(shape=(1, 4), dtype=float)

y_pred = np.zeros_like(a=0, shape=(1, 4), dtype=float)

w = np.random.rand(1, 4)
b = np.random.rand(1, 4)

for i in range(epoch):    
    prediction = np.dot(X, w.T) + b   
    
    for j in range(n):                        
        if y[j] > prediction[0][j]:
            w[0][j] += 0.5            
        else:
            w[0][j] -= 0.5
            
    #print(w)
    

    print(prediction)      
              
        
        
# plt.plot(x, y)
# plt.plot(x, prediction[0])
# plt.show()
