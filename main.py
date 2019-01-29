# Function created on October 17th, 2016 by Eric Braun 10121660
# This function creates a neural network with fixed inputs of 64 and outputs of 10
# It will take in data from training.txt to train the neural network using
# back propagation then test the weight values using testing.txt
#
# Function created for CISC 452 Assignment 2



from Neural_Network import Neural_Network
import numpy as np
np.seterr(all = 'ignore')

def main():

    # These values can be changed to increase/decrease time & accuracy
    hidden_nodes = 25
    iterations = 20
    learning_rate = 1
    momentum = 0.5

    # Create a neural network
    N = Neural_Network(hidden_nodes, iterations, learning_rate, momentum)

    # Read in the datasets
    Train = read_in('training.txt')
    Test = read_in('testing.txt')

    N.train_network(Train) # Train the neural network using the training dataset
    N.test_network(Test) # Test the neural network using the testing dataset

# Function to read data in
def read_in(data):
    data = np.loadtxt(data, delimiter = ',')

    # Last integer in the row is the actual value
    y = data[:,data.shape[1]-1]

    data = data[:,:data.shape[1]-1] # x data
    data -= data.min() # scale the data so values are between 0 and 1
    data /= data.max() # scale

    out = []

    # populate the tuple list with the data
    for i in range(data.shape[0]):
        T = more_outputs(y[i])
        output = list((data[i,:].tolist(), T))
        out.append(output)
    return out

# Create a vector of length 10 for each output from the integer value
def more_outputs(x):
    output = int(x) # Convert float into integer
    vector_outputs = np.zeros(10)
    vector_outputs[output] = 1
    return vector_outputs

if __name__ == '__main__':
    main()
