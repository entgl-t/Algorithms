#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from scipy.optimize import minimize



def variational_ansatz_zero(params, wires):

    n_qubits = len(wires)
    D1 = [0,1]
    D2 = [i for i in range(len(D1) ,params.shape[0]-1)]
    depth = params.shape[0]
    #n_rotations = len(D1)

    qml.BasisState(np.array([0 for i in range(n_qubits)]), wires=[i for i in range(n_qubits)])
    for d in D1:
        for i in range(n_qubits//2, n_qubits):

            qml.RY(params[d,i],wires=[i]) 
            qml.RZ(params[d,i],wires=[i])
    for d2 in D2:
        for i in range(n_qubits):
            qml.RY(params[d2,i],wires=[i]) 
            qml.RZ(params[d2,i],wires=[i])             

        for i in range(n_qubits//2):

            qml.CZ(wires=[2*i, 2*i + 1])   #circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubits//2 - 1):
            qml.CZ(wires=[2*i+1, 2*i + 2])   #circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubits):
        qml.RY(params[depth-1,i],wires=[i]) 
        qml.RZ(params[depth-1,i],wires=[i])



def variational_ansatz_one(params, wires):

    n_qubits = len(wires)
    D1 = [0,1]
    D2 = [i for i in range(len(D1) ,params.shape[0]-1)]
    depth = params.shape[0]
    #n_rotations = len(D1)

    qml.BasisState(np.array([1] + [0 for i in range(n_qubits-1)]), wires=[i for i in range(n_qubits)])
    for d in D1:
        for i in range(n_qubits//2, n_qubits):

            qml.RY(params[d,i],wires=[i]) 
            qml.RZ(params[d,i],wires=[i])
    for d2 in D2:
        for i in range(n_qubits):
            qml.RY(params[d2,i],wires=[i]) 
            qml.RZ(params[d2,i],wires=[i])             

        for i in range(n_qubits//2):

            qml.CZ(wires=[2*i, 2*i + 1])   #circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubits//2-1):
            qml.CZ(wires=[2*i+1, 2*i + 2])   #circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubits):
        qml.RY(params[depth-1,i],wires=[i]) 
        qml.RZ(params[depth-1,i],wires=[i])





def variational_ansatz_two(params, wires):

    n_qubits = len(wires)
    D1 = [0,1]
    D2 = [i for i in range(len(D1) ,params.shape[0]-1)]
    depth = params.shape[0]
    #n_rotations = len(D1)

    qml.BasisState(np.array([0,1] + [0 for i in range(n_qubits-2)]), wires=[i for i in range(n_qubits)])
    for d in D1:
        for i in range(n_qubits//2, n_qubits):

            qml.RY(params[d,i],wires=[i]) 
            qml.RZ(params[d,i],wires=[i])
    for d2 in D2:
        for i in range(n_qubits):
            qml.RY(params[d2,i],wires=[i]) 
            qml.RZ(params[d2,i],wires=[i])             

        for i in range(n_qubits//2):

            qml.CZ(wires=[2*i, 2*i + 1])   #circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubits//2-1):
            qml.CZ(wires=[2*i+1, 2*i + 2])   #circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubits):
        qml.RY(params[depth-1,i],wires=[i]) 
        qml.RZ(params[depth-1,i],wires=[i])
        


   

########################################################################################



def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #

    # Initialize parameters
    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1
    #np.random.seed(0)
    #params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    params = np.random.uniform(low=0, high=2*np.pi, size=(6,num_qubits))
    weights = [1,1,1]#np.random.uniform(0, 1, size=(3,))
    print(weights)

    energy = 0

    # QHACK #

    # Create a quantum device, set up a cost funtion and optimizer, and run the VQE.
    # (We recommend ~500 iterations to ensure convergence for this problem,
    # or you can design your own convergence criteria)

    # QHACK #
    dev = qml.device('default.qubit', wires = num_qubits)
    
    @qml.qnode(dev)
    def testt(n_qubits):
      qml.BasisState(np.array([0,1] + [0 for i in range(n_qubits-2)]), wires=[i for i in range(n_qubits)])
      return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    print('testddddddddd', testt(num_qubits))
    

    # Minimize the circuit
    #opt = qml.GradientDescentOptimizer(stepsize=0.4)
    opt = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.99, eps=1e-08)
    #opt = qml.MomentumOptimizer(stepsize=0.1, momentum=0.99)

    steps = 600

    #cost_fn_zero = qml.ExpvalCost(variational_ansatz_zero, H, dev)
    cost_fn_zero = qml.ExpvalCost(variational_ansatz_zero, H, dev)
    cost_fn_one = qml.ExpvalCost(variational_ansatz_one, H, dev)
    cost_fn_two = qml.ExpvalCost(variational_ansatz_two, H, dev)
    


    def cost(params):
        total = cost_fn_zero(params)*weights[0] + cost_fn_one(params)*weights[1] + cost_fn_two(params)*weights[2]
        return total
    
    def cost_fn(*qnode_args, **qnode_kwargs):
                """Combine results from grouped QNode executions with grouped coefficients"""
                total = 0
                total = cost_fn_zero(*qnode_args, **qnode_kwargs)*weights[0] + cost_fn_one(*qnode_args, **qnode_kwargs)*weights[1] + cost_fn_two(*qnode_args, **qnode_kwargs)*weights[2]
                return total
    '''
    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}
    opt = minimize(cost, params,
               method=method,
               callback=callback)
    '''
    for i in range(steps):
        
        params, prev_energy_total= opt.step_and_cost(cost_fn, params)
        total_energy = cost_fn(params)
        conv = np.abs(total_energy - prev_energy_total)
       
        if i % 4 == 0:
           print("Iteration = {:},  E = {:.8f} Ha".format(i, total_energy))

        if conv <= 1e-06:
            break

    


   
    energies[0] = cost_fn_zero(params)
    energies[1] = cost_fn_one(params)
    energies[2] = cost_fn_two(params)
    
    # QHACK #

    return sorted(energies)


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
