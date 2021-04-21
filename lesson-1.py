import sys
import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import math

e = math.e

output_layers = [4.8, 1.21, 2.385]
exp_values = np.exp(output_layers)
s = np.sum(exp_values, axis=0)
norm_values = exp_values/s
# print(exp_values)
# print(norm_values)
# print(sum(norm_values))

nnfs.init()
inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0.1, 0.3, -0.5, 1],
            [0.5, -0.9, 0.2, -0.5],
            [-0.2, -0.2, 0.1, 0.8]]
biases1 = [2, 3, 0.5]

weights2 = [[0.2, 0.8, -0.5],
            [0.5, -0.91, 0.26],
            [-0.26, -0.27, 0.17]]

biases2 = [2, 3, 0.5]

# layer1 = np.dot(inputs, np.array(weights1).T) + biases1
# layer2 = np.dot(layer1, np.array(weights2).T) + biases2

x, y = spiral_data(samples=100, classes=3)
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='brg')
plt.show()

        




    

