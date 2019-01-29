# Function created on October 17th, 2016 by Eric Braun 10121660 13eb20
# This file defines the functions needed to run main.py
#
# Contains the class of Neural_Network which takes in parameters of hidden nodes, number of iterations, learning rate, and momentum
# Will use the training data to train the network using feed_forward and back_propagation functions
# With the weights created, it will test to see how many numbers it can accurately categorize using testing.txt data
#
# File created for CISC 452, Assignment 2

import numpy as np
import math
import random

class Neural_Network(object):
    # Contructor
    def __init__(self, hidden, iterations, learning_rate, momentum):

        # parameters
        self.hidden = hidden
        self.activation_hidden = [1.0] * self.hidden
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum

        # arrays to be used for activations
        self.activation_input = [1.0] * 65
        self.activation_output = [1.0] * 10

        # create running array of changing weight values
        self.c_input = np.zeros((65, self.hidden))
        self.c_output = np.zeros((self.hidden, 10))

        # create randomized weights using range of input and outputs
        input_range = 1.0 / 65 ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.input_weights = np.random.normal(loc = 0, scale = input_range, size = (65, self.hidden))
        self.output_weights = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, 10))
    # End constructor

    def feed_forward(self, activations):

        # fill in input layer
        for i in range(64): # 1 less than the activations to avoid
            self.activation_input[i] = activations[i]

        # calculate the hidden activations by calculating the sum of their inputs * weights
        for j in range(self.hidden):
            sum = 0.0
            for i in range(65):
                sum += self.activation_input[i] * self.input_weights[i][j]
            self.activation_hidden[j] = sigmoid(sum)

        # calculate the output activations from the sum of the hidden layer outputs * out put weights
        for k in range(10):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.activation_hidden[j] * self.output_weights[j][k]
            self.activation_output[k] = sigmoid(sum)
        return self.activation_output[:]
    # End feed_forward function

    def back_propagation(self, targets):

        # calculate error terms for output layer
        # The error is filled into output_change which tells which direction the weight needs to be changed
        output_direction = [0.0] * 10
        for k in range(10):
            error = -(targets[k] - self.activation_output[k]) # (d-y)
            output_direction[k] = sigmoid_derivative(self.activation_output[k]) * error

        # calculate error terms for hidden layer
        hidden_direction = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(10):
                error += output_direction[k] * self.output_weights[j][k] # error is activation from previous layer
            hidden_direction[j] = sigmoid_derivative(self.activation_hidden[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(10):
                change = output_direction[k] * self.activation_hidden[j]
                self.output_weights[j][k] -= self.learning_rate * change + self.c_output[j][k] * self.momentum
                self.c_output[j][k] = change

        # update the weights connecting input to hidden
        for i in range(65):
            for j in range(self.hidden):
                change = hidden_direction[j] * self.activation_input[i]
                self.input_weights[i][j] -= self.learning_rate * change + self.c_input[i][j] * self.momentum
                self.c_input[i][j] = change

        # calculate the error using root mean square error
        errors = np.zeros(len(targets))
        for k in range(len(targets)):
            errors[k] = (targets[k] - self.activation_output[k]) ** 2
        T = np.mean(errors)
        error = math.sqrt(T)
        return error
    # End back_propagation function

    def train_network(self, train_data):
        for i in range(self.iterations): # for each iteration
            random.shuffle(train_data) # shuffle the data to make more random
            error = 0.0
            for j in train_data:
                self.feed_forward(j[0]) # Feed inputs through network
                error += self.back_propagation(j[1]) # bring error back through network using targets
            print('Error: %-.5f' % error) # print the error after every iteration
    # End train_network function

    def test_network(self, test_data):
        length = len(test_data)
        runningCorrect = 0 # running correct sum to calculate accuracy
        for i in test_data:
            temp = self.feed_forward(i[0]) # pass through feed forward network to create calculated data
            calculated = transform_prediction(temp) # convert 10 element vector into corresponding integer value

            # Calculate accuracy
            actual = transform_prediction(i[1])
            if (actual == calculated):
                runningCorrect += 1
            print('Actual:', actual, 'calculated:', calculated)
        print('Correcly sorted', runningCorrect, 'out of', length)
        accuracy = runningCorrect / length
        print('accuracy:', accuracy)
    # End test_network function

# End Neural_Network class

# The output sigmoid function used for feed forward propagation
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

# The derivative of the sigmoid output function used for back propagation
def sigmoid_derivative(value):
    return value * (1.0 - value)

# Function to transform the predicted vector into the closest integer value
def transform_prediction(x):
    for i in range(len(x)):
        if x[i] == max(x):
            return i
