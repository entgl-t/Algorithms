#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np



def variational_ansatz_zero(params, wires):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    This is ansatz is used to help with the problem structure. It applies
    alternating layers of rotations and CNOTs.

    Don't worry about the contents of this function for now—you'll be designing
    your own ansatze in a later problem.

    Args:
        params (np.ndarray): An array of floating-point numbers with size (n, 3),
            where n is the number of parameter sets required (this is determined by
            the problem Hamiltonian).
        wires (qml.Wires): The device wires this circuit will run on.
    """
    n_qubits = len(wires)
    n_rotations = len(params)

    qml.BasisState(np.array([0 for i in range(n_qubits)]), wires=[i for i in range(n_qubits)])
    if n_rotations > 1:
        n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits
 
        # Alternating layers of unitary rotations on every qubit followed by a
        # ring cascade of CNOTs.

        
        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

        # There may be "extra" parameter sets required for which it's not necessarily
        # to perform another full alternating cycle. Apply these to the qubits as needed.
        extra_params = params[-n_extra_rots:, :]
        extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
        qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
    else:
        # For 1-qubit case, just a single rotation to the qubit
        qml.Rot(*params[0], wires=wires[0])



def variational_ansatz_one(params, wires):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    This is ansatz is used to help with the problem structure. It applies
    alternating layers of rotations and CNOTs.

    Don't worry about the contents of this function for now—you'll be designing
    your own ansatze in a later problem.

    Args:
        params (np.ndarray): An array of floating-point numbers with size (n, 3),
            where n is the number of parameter sets required (this is determined by
            the problem Hamiltonian).
        wires (qml.Wires): The device wires this circuit will run on.
    """
    n_qubits = len(wires)
    n_rotations = len(params)

    qml.BasisState(np.array([1] + [0 for i in range(n_qubits-1)]), wires=[i for i in range(n_qubits)])
    if n_rotations > 1:
        n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits

        # Alternating layers of unitary rotations on every qubit followed by a
        # ring cascade of CNOTs.
        #for i in range(0,int(n_qubits/2)):
        #qml.PauliX(n_qubits - 1)

       
        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

        # There may be "extra" parameter sets required for which it's not necessarily
        # to perform another full alternating cycle. Apply these to the qubits as needed.
        extra_params = params[-n_extra_rots:, :]
        extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
        qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
    else:
        # For 1-qubit case, just a single rotation to the qubit
        qml.Rot(*params[0], wires=wires[0])





def variational_ansatz_two(params, wires):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    This is ansatz is used to help with the problem structure. It applies
    alternating layers of rotations and CNOTs.

    Don't worry about the contents of this function for now—you'll be designing
    your own ansatze in a later problem.

    Args:
        params (np.ndarray): An array of floating-point numbers with size (n, 3),
            where n is the number of parameter sets required (this is determined by
            the problem Hamiltonian).
        wires (qml.Wires): The device wires this circuit will run on.
    """
    n_qubits = len(wires)
    n_rotations = len(params)

    qml.BasisState(np.array([0,1] + [0 for i in range(n_qubits-2)]), wires=[i for i in range(n_qubits)])
    if n_rotations > 1:
        n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits

        # Alternating layers of unitary rotations on every qubit followed by a
        # ring cascade of CNOTs.
        #for i in range(0,n_qubits):
        
        
        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

        # There may be "extra" parameter sets required for which it's not necessarily
        # to perform another full alternating cycle. Apply these to the qubits as needed.
        extra_params = params[-n_extra_rots:, :]
        extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
        qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
    else:
        # For 1-qubit case, just a single rotation to the qubit
        qml.Rot(*params[0], wires=wires[0])




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
    np.random.seed(0)
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    #params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_qubits-1,))
    weights = [1,1,1]#np.random.uniform(0, 1, size=(3,))
    print(weights)

    energy = 0

    # QHACK #

    # Create a quantum device, set up a cost funtion and optimizer, and run the VQE.
    # (We recommend ~500 iterations to ensure convergence for this problem,
    # or you can design your own convergence criteria)

    # QHACK #
    dev = qml.device('default.qubit', wires = num_qubits)

    

    # Minimize the circuit
    #opt = qml.GradientDescentOptimizer(stepsize=0.4)
    opt = qml.AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.99, eps=1e-08)
    #opt = qml.MomentumOptimizer(stepsize=0.1, momentum=0.99)

    steps = 600

    #cost_fn_zero = qml.ExpvalCost(variational_ansatz_zero, H, dev)
    cost_fn_zero = qml.ExpvalCost(variational_ansatz_zero, H, dev)
    cost_fn_one = qml.ExpvalCost(variational_ansatz_one, H, dev)
    cost_fn_two = qml.ExpvalCost(variational_ansatz_two, H, dev)
    #total_cost = weights[0]*cost_fn_zero + weights[1]*cost_fn_one
    #total_cost = np.sum(cost_fn_zero , cost_fn_one)
    
    def cost_fn(*qnode_args, **qnode_kwargs):
                """Combine results from grouped QNode executions with grouped coefficients"""
                total = 0
                total = cost_fn_one(*qnode_args, **qnode_kwargs)
                #total = cost_fn_zero(*qnode_args, **qnode_kwargs)*weights[0] + cost_fn_one(*qnode_args, **qnode_kwargs)*weights[1] + cost_fn_two(*qnode_args, **qnode_kwargs)*weights[2]
                return total

    for i in range(steps):
        
        params, prev_energy_total= opt.step_and_cost(cost_fn, params)
        total_energy = cost_fn(params)
        conv = np.abs(total_energy - prev_energy_total)
       
        if i % 4 == 0:
           print("Iteration = {:},  E = {:.8f} Ha".format(i, total_energy))

        if conv <= 1e-06:
            break




   
    #energies[0] = cost_fn_zero(params)
    energies[1] = cost_fn_one(params)
    #energies[2] = cost_fn_two(params)
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
