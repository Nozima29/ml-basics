# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:02:30 2021

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2
  
x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approx_deriv = (y2 - y1) / (x2 - x1)
b = y2 - approx_deriv*x2

def tangent(x):
    return approx_deriv*x + b 


to_plot = [x1-1, x1, x1+1, x1+2]
plt.plot(to_plot, [tangent(i) for i in to_plot])
plt.show()
print([tangent(i) for i in to_plot])


