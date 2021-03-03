#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    # Gradients #
    forward = np.zeros([5], dtype=np.float64)
    backward = np.zeros([5], dtype=np.float64)
    for i in range(len(weights)):
        shifted = weights.copy()
        shifted[i] = weights[i] + np.pi/2
        forward[i] = circuit(shifted) 
          
        shifted[i] = weights[i] -  np.pi/2
        backward[i] = circuit(shifted) 
            
        gradient[i] = (forward[i] - backward[i])/(2*np.sin(np.pi/2))
   
   
    # Hessian #
    org= circuit(weights) 
    for i in range(len(weights)):
        for j in range(i , len(weights)):
        
            shifted2 = weights.copy()
            if i == j:
               hessian[i,j] = (forward[i]  - 2*org + backward[i] )/(2*np.sin(np.pi/2))
               

               #shifted2[i] = weights[i] + np.pi
               #forward1 = circuit(shifted2) 
               #hessian[i,j] = (forward1  - org )/(2*np.sin(np.pi/2))
                
            else:
                shifted2[i] = weights[i] + np.pi/4
                shifted2[j] = weights[j] + np.pi/4
                forward1 = circuit(shifted2) 
          
                shifted2[i] = weights[i] -  np.pi/4
                shifted2[j] = weights[j] +  np.pi/4
                backward1 = circuit(shifted2) 

                shifted2[i] = weights[i] +  np.pi/4
                shifted2[j] = weights[j] -  np.pi/4
                forward2 = circuit(shifted2)         

                shifted2[i] = weights[i] -  np.pi/4
                shifted2[j] = weights[j] -  np.pi/4
                backward2 = circuit(shifted2)

                
                
                hessian[i,j] = (forward1  - backward1 - forward2 + backward2 )/((2*np.sin(np.pi/4))**2)
                hessian[j,i] = hessian[i,j]
            
           

    # QHACK #
    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
