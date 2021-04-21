import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import math
#nnfs.init() 

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
target_output = [1, 0, 0]
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

confidence = np.sum(softmax_outputs * class_targets, axis=1)
loss = np.mean(-np.log(confidence))


print(confidence)
print(loss)
