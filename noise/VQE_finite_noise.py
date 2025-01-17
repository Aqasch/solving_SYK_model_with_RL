from qulacs import ParametricQuantumCircuit, QuantumState, DensityMatrix
from qulacs.gate import CNOT
from qulacs.gate import BitFlipNoise, DephasingNoise, IndependentXZNoise, DepolarizingNoise, TwoQubitDepolarizingNoise
from qulacs.gate import *
import numpy as np
from utils_noise import *
from qulacsvis import circuit_drawer
from copy import deepcopy



class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.noise_models = noise_models
        self.noise_values = noise_values
        self.ansatz = ParametricQuantumCircuit(n_qubits)
        self.action_dict = dictionary_of_actions(self.n_qubits)

    def construct_ansatz(self, state):
        

        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]


            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))
                    # decoherence0 = AmplitudeDampingNoise(ctrl[r], damping_rate)
                    # decoherence1 = AmplitudeDampingNoise(targ[r], damping_rate)
                    gate = TwoQubitDepolarizingNoise(ctrl[r], targ[r], 8.043e-3)
                    self.ansatz.add_gate(gate)
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]


            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    
                    if r == 0:
                        self.ansatz.add_parametric_RX_gate(rot_qubit, thetas[0][rot_qubit])
                        gate = BitFlipNoise(rot_qubit, 2.342e-4)
                        self.ansatz.add_gate(gate)
                    elif r == 1:
                        self.ansatz.add_parametric_RY_gate(rot_qubit, thetas[1][rot_qubit])
                        gate = BitFlipNoise(rot_qubit,2.342e-4)
                        self.ansatz.add_gate(gate)
                    elif r == 2:
                        self.ansatz.add_parametric_RZ_gate(rot_qubit,  thetas[2][rot_qubit])
                        gate = BitFlipNoise(rot_qubit, 2.342e-4)
                        self.ansatz.add_gate(gate)
                    else:
                        print('The angle is not right')
        
        return self.ansatz


def get_free_energy_qulacs(angles, observable, 
                      circuit, n_qubits, inverse_temp,
                      which_angles=[]):
    """"
    Function for energy minimization using Qulacs

    Output:
    free energy [float] : free energy value 
    
    """

    # print(circuit_drawer(circuit))
        
    parameter_count_qulacs = circuit.get_parameter_count()
    
    if not list(which_angles):
        which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])
    
    free_energy, _, _ , _= get_free_energy(n_qubits,circuit,observable, inverse_temp,)
    return free_energy

def get_free_energy(n_qubits,circuit,op,inverse_temp):


    entropy_circ_gate_count = 3*n_qubits+n_qubits
    total_gate_count = circuit.get_gate_count()

    
    """
    entropy calculation
    """
    # MAKE ENTOPY CIRCUIT
    state = QuantumState(n_qubits)
    gate_parsed_list_entropy = []
    for i in range(entropy_circ_gate_count):
        gate_parsed_list_entropy.append(circuit.get_gate(i))
    entropy_circuit = ParametricQuantumCircuit(n_qubits)
    for i in range(entropy_circ_gate_count):
        entropy_circuit.add_gate(gate_parsed_list_entropy[i])

    entropy_circuit.update_quantum_state(state)  # Update the quantum state with the circuit
    statevector = state.get_vector()  # Get the statevector of the quantum state
    
    

    # Calculate the probabilities from the statevector
    probabilities = np.abs(statevector)**2  # Probabilities are the squared magnitudes of the amplitudes
    entropy = -np.sum(probabilities * np.log(probabilities+1e-10))


    # MAKE EXPVAL CIRCUIT
    expval = 0
    gate_parsed_list_expval = []
    for i in range(entropy_circ_gate_count, total_gate_count):
        gate_parsed_list_expval.append(circuit.get_gate(i))
    expval_circuit = ParametricQuantumCircuit(n_qubits)
    for j in range(len(range(entropy_circ_gate_count, total_gate_count))):
        expval_circuit.add_gate(gate_parsed_list_expval[j])

    # """
    # expval calculation
    # """

    # """
    # original
    # """
    state_as_expval_circ_initial = np.diag(probabilities)
    state_to_feed = DensityMatrix(n_qubits)
    state_to_feed.load(state_as_expval_circ_initial)
    expval_circuit.update_quantum_state(state_to_feed)
    expval_circuit.update_quantum_state(state_to_feed)
    rho_fidelity = state_to_feed.get_matrix()
    # expval += np.trace(rho_fidelity @ op).real


    ###
    """ TESTING!!! """
    expval_circuit.update_quantum_state(state)
    final_statevector = state.get_vector()
    rho = np.outer(final_statevector, np.conjugate(final_statevector))
    expval = np.trace(rho @ op).real
    """ TESTING!!! """
    ###

    



    """
    the cost function
    """
    free_energy = expval - (1/inverse_temp)*entropy

    return free_energy, expval, entropy, rho_fidelity


if __name__ == "__main__":
    pass


















