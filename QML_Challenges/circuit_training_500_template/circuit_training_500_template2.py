#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.templates import AmplitudeEmbedding


dev = qml.device('default.qubit', wires = 2)
'''
def get_angles(x):

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def statepreparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)
'''
def layer(W):

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    #qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    #qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    #qml.CNOT(wires=[1, 0])
    #qml.CNOT(wires=[2, 3])
    #qml.CNOT(wires=[3, 0])


def statepreparation(x):
    #qml.BasisState(x, wires=[0, 1])
    AmplitudeEmbedding(features=x, wires=range(2), normalize=True)
    #print("amplitude vector: ", np.real(dev.state))


@qml.qnode(dev)
def circuit(weights, angles):

    statepreparation(angles)
   
    for W in weights:
        layer(W)

    #return qml.probs(wires=[0, 1])
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
    #print("First X sample (normalized):", X_norm[0])


    # angles for state preparation are new features
    features = X_train #features = np.array([get_angles(x) for x in X_norm])
    features_test = X_test  #np.array([get_angles(x) for x in X_test_norm])
    
    #labels
    #for -1 prob vector [0,0,1,0]
    #for 0 prob vector [0,1,0,0]
    #for 1 prob vector [1,0,0,0]

    
    #opt = NesterovMomentumOptimizer(0.5)
    #opt = qml.GradientDescentOptimizer(stepsize=0.4)
    opt = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.99, eps=1e-08)
    #opt = qml.MomentumOptimizer(stepsize=0.1, momentum=0.99)
    batch_size = 5
    


    var = var_init
    for it in range(150):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = features[batch_index]
        Y_batch = Y_train[batch_index]
        #print("Hooooooooooooooooooooooooooooo")
        var = opt.step(lambda v: cost(v, X_batch, Y_batch), var)            
        
   
        # Compute accuracy
        
        predictions_train = []
        
        for x in features:
            res = variational_classifier(var, x)
            if res<1/3 and res>-1/3:
                predictions_train.append(0)
            elif res>1/3:
                predictions_train.append(1)
            elif res<-1/3:
                predictions_train.append(-1)
        
        #predictions_train = [np.sign(variational_classifier(var, x)) for x in features]
        acc = accuracy(Y_train, predictions_train)
        
        print(
            "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                it + 1, cost(var, features, Y_train), acc
            )
        )
        
        
    predictions = []
        
    for x in features_test:
        res = variational_classifier(var, x)
        if np.argmax(res) == 0:
            predictions.append(1)
        elif np.argmax(res) == 1:
            predictions.append(0)
        elif np.argmax(res) == 2:
            predictions.append(-1)
             
    test_ans = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]
    print(accuracy(test_ans, predictions), 'Here')
    # QHACK #
    with open('weightsss.txt', 'w') as f:
        for item in var:
            f.write("%s\n" % item)

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
