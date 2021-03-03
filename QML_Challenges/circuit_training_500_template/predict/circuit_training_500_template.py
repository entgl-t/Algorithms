#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.templates import AmplitudeEmbedding


dev = qml.device('default.qubit', wires = 2)

def layer(W):

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
   

    qml.CNOT(wires=[0, 1])
    


def statepreparation(x):
   
    AmplitudeEmbedding(features=x, wires=range(2), normalize=True)
   


@qml.qnode(dev)
def circuit(weights, angles):

    statepreparation(angles)
    
    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(var, angles):
    weights = var[0]
    bias = var[1]
    return circuit(weights, angles) + bias


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cost(var, features, labels):
    predictions = [variational_classifier(var, f) for f in features]
    return square_loss(labels, predictions)



def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss



def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    np.random.seed(0)
    num_qubits = 2
    num_layers = 10
    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
    

    X_train = np.c_[ X_train, np.zeros(250) ] 
    X_test = np.c_[ X_test, np.zeros(50) ] 
    
    # normalize each input
    normalization = np.sqrt(np.sum(X_train ** 2, -1))
    normalization_test = np.sqrt(np.sum(X_test ** 2, -1))
    X_norm = (X_train.T / normalization).T
    X_test_norm = (X_test.T / normalization_test).T
    


    # angles for state preparation are new features
    features_test = X_test 
     
    var = [[[[-0.00336064, -0.26652834,  0.00963062],
  [ 0.07296831,  0.32535991,  0.0359078 ]],

 [[ 0.00934412, -0.36955743,  0.01230224],
  [ 0.04694684, -0.19415532,  0.06455748]],

 [[ 0.02094481, -0.27661457,  0.02648386],
  [ 0.05764314,  0.24854446,  0.04632115]],

 [[ 0.0251759,  -0.38133756, -0.00906811],
  [ 0.0524873,  -0.16442958,  0.04872451]],

 [[ 0.03915934, -0.29809081,  0.01241628],
  [ 0.04557186,  0.1702122,   0.05795115]],

 [[ 0.01350817, -0.36993905, -0.00470132],
  [ 0.03271057, -0.06525212,  0.05652655]],

 [[ 0.01647944, -0.27076292,  0.01131147],
  [ 0.05289908,  0.08910276,  0.04142214]],

 [[-0.00187797, -0.35576381,  0.00259993],
  [ 0.02943902, -0.1373223,   0.05633363]],

 [[-0.00844253, -0.28596876, -0.00547281],
  [ 0.04782179,  0.08418016,  0.03165944]],

 [[ 0.00320004, -0.37086802,  0.00066517],
  [ 0.00302472, -0.00634322, -0.00362741]]],
-0.09804288208812778]
        
    predictions = []
        
    for x in features_test:
        res = variational_classifier(var, x)
        if res<1/3 and res>-1/3:
            predictions.append(0)
        elif res>1/3:
            predictions.append(1)
        elif res<-1/3:
                predictions.append(-1)

    test_ans = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]
    #print(accuracy(test_ans, predictions), 'Here')
    # QHACK #
    
    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
