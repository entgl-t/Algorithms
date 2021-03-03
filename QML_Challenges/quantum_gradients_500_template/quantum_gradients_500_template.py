#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    #@qml.qnode(dev)
    
    

   
    N=len(params)
    gradient = np.zeros([N], dtype=np.float64)
    Fubini = np.zeros([N, N])

    for i in range(N):


        for j in range(N):

                params2_1 = params.copy()
                params2_2 = params.copy()
                params2_3 = params.copy()
                params2_4 = params.copy() 
        
                params2_1[i] += np.pi/2
                params2_2[i] += np.pi/2
                params2_3[i] -= np.pi/2
                params2_4[i] -= np.pi/2
     
                params2_1[j] +=  np.pi/2       
      	  
                params2_2[j] -= np.pi/2
    
                params2_3[j] += np.pi/2    
                params2_4[j] -= np.pi/2
                Fubini[i,j] = (1/8)*(-inner_prod(params,params2_1) + inner_prod(params,params2_2) + inner_prod(params,params2_3) - inner_prod(params,params2_4))
   

    
    test = qml.metric_tensor(qnode)
    res_test = test(params)
    
   
    #Gradient
    for i in range(N):
        shifted = params.copy()
        shifted[i] = params[i] + np.pi/2
        forward = qnode(shifted) 
          
        shifted[i] = params[i] -  np.pi/2
        backward = qnode(shifted) 
            
        gradient[i] = (forward - backward)/(2*np.sin(np.pi/2))
        
        
    Fubini_inv = np.linalg.inv(Fubini)
    test3 = np.linalg.pinv(res_test)
    test3 = test3.dot(gradient)
    res_test_inv = np.linalg.inv(res_test)
    natural_grad = Fubini_inv.dot(gradient)
    
    # QHACK #

    return natural_grad


@qml.qnode(dev)
def back_to_org(params1,params2):
        
    
        variational_circuit(params2)
        qml.inv(variational_circuit2(params1))
         
        return qml.probs(wires=[0, 1, 2])

@qml.template
def variational_circuit2(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)



#@qml.qnode(dev)

def inner_prod(params1,params2):
    
    res =  back_to_org(params1,params2)  
    return  res[0]


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)




def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
