# -*- coding: utf-8 -*-
"""
Simple Neural Network w/ no hidden layers

@author: Matthew Chen
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# The training set, with 4 examples consisting of 3
# input values and 1 output value
training_inputs = np.array([[0,0,1],
                             [1,1,1],
                             [1,0,1],
                             [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T # <-- Transpose of the matrix/array
# training_outputs looks like this:
# [0
#  1
#  1
#  0]

# Seed the random number generator
np.random.seed(1) 

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for iteration in range(20000):
    # Pass training set through the neural network
    input_layer = training_inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calculate the error rate
    error = training_outputs - outputs

    # Multiply error by input and gradient of the sigmoid function
    # Less confident weights are adjusted more through the nature of the function
    adjustments = error * sigmoid_derivative(outputs)
    
    # Adjust synaptic weights
    synaptic_weights += np.dot(input_layer.T, adjustments)
    
print("Synaptic weights after training: ")
print(synaptic_weights)

print("Outputs after training: ")
print(outputs)
    
    
    
