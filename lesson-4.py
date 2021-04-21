# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:49:59 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data, spiral_data
from Main import DenseLayer, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy

nnfs.init()

x, y = spiral_data(samples=100, classes=3)
dense1 = DenseLayer(2, 3)
act1 = Activation_ReLU()
dense2 = DenseLayer(3, 3)
act2 = Activation_Softmax()
loss_func = Loss_CategoricalCrossentropy()
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()

best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(1000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)
    
    dense1.forward(x)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    act2.forward(dense2.output)
    
    loss = loss_func.calculate(act2.output, y)
    
    predictions = np.argmax(act2.output, axis=1)
    accuracy = np.mean(predictions==y)
    
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
    
    
    
# plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='brg')
# plt.show()

