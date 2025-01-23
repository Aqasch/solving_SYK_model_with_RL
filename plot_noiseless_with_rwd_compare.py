import os
import json
import numpy as np
import scipy.linalg as la
from itertools import product
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import matplotlib.lines as mlines
from noise.true_helmolhz_free_energy import helmholtz_free_energy

os.environ["XDG_SESSION_TYPE"] = "xcb"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16

})

# Set up matplotlib for LaTeX rendering
def state_fidelity(rho, sigma):
    """
    Calculate the fidelity between two quantum states.
    
    Parameters:
    - rho: numpy array, density matrix of state 1
    - sigma: numpy array, density matrix of state 2
    
    Returns:
    - Fidelity (float)
    """
    # Ensure the matrices are Hermitian
    assert np.allclose(rho, rho.conj().T), "rho is not Hermitian"
    assert np.allclose(sigma, sigma.conj().T), "sigma is not Hermitian"
    
    # Compute sqrt(rho)
    sqrt_rho = la.sqrtm(rho)
    
    # Compute sqrt(sqrt(rho) * sigma * sqrt(rho))
    inner_product = la.sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    
    # Calculate fidelity
    fidelity = np.trace(inner_product)
    return np.real(fidelity)


def dictionary_of_actions(num_qubits):
    dictionary = dict()
    i = 0
    # Generate CNOT gate actions
    for c, x in product(range(num_qubits), range(1, num_qubits)):
        # c: control qubit, x: target qubit offset
        dictionary[i] = [c, x, num_qubits, 0]  # [control, offset, num_qubits, rotation_axis]
        i += 1
    # Generate rotation gate actions
    for r, h in product(range(num_qubits), range(1, 4)):
        # r: rotation qubit, h: rotation axis (1=X, 2=Y, 3=Z)
        dictionary[i] = [num_qubits, 0, r, h]  # [num_qubits, 0, rotation_qubit, rotation_axis]
        i += 1
    return dictionary

def make_circuit_qiskit(action, qubits, circuit):
    ctrl = action[0]
    targ = (action[0] + action[1]) % qubits  # Calculate target qubit for CNOT
    rot_qubit = action[2]
    rot_axis = action[3]

    # Apply CNOT gate if control qubit is valid
    if ctrl < qubits:
        circuit.cx([ctrl], [targ])
    
    # Apply rotation gate if rotation qubit is valid
    if rot_qubit < qubits:
        if rot_axis == 1:
            circuit.rx(0, rot_qubit)  # Rotation around X-axis
        elif rot_axis == 2:
            circuit.ry(0, rot_qubit)  # Rotation around Y-axis
        elif rot_axis == 3:
            circuit.rz(0, rot_qubit)  # Rotation around Z-axis
    
    return circuit


def get_user_choice(question):
    while True:
        choice = input(question + " (yes/no): ").lower()
        if choice in ['yes', 'no']:
            if choice == 'yes':
                return 1
            elif choice == 'no':
                return 0

        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

error_only = get_user_choice("Do you want to plot with only error reward?")
if error_only == 0:
    fidelity_only = get_user_choice("Do you want to plot with only fidelity reward?")
    if fidelity_only == 0:
        both = get_user_choice("Do you want to plot with both error and fidelity reward?")


if error_only:
    print('Plotting with only error reward')
    fidelity_eng_list = [0]
elif fidelity_only:
    print('Plotting with only fidelity reward')
    fidelity_eng_list = [1]
elif both:
    print('Plotting with both error and fidelity reward')
    fidelity_eng_list = [0, 1]
else:
    print('No plot option selected')
    fidelity_eng_list = []

# The rest of your code continues here

fig, ax = plt.subplots(ncols=1, nrows=3, figsize = (5,8), sharex=True)
seed = 1

def get_user_choice_qubits(question):
    while True:
        choice = input(question + " (4 or 5 or 6 or 7): ").lower()
        if choice in ['4', '5', '6', '7']:
            return int(choice)

        else:
            print("Invalid input. Please enter at least a qubit number.")

neural_net_list = ['3D_CNN', '1D_FNN']
neural_net = neural_net_list[0]
inverse_temp_list = ['5p2',18, 35]

qubits = get_user_choice_qubits("Do you want the plot for how many qubits?")

if qubits == 4:
    layer = 5
elif qubits == 5:
    layer =3
else:
    layer = 4

label_list = ['Free energy reward', 'Free energy and fidelity reward']
marker_list = ['o', 'x']
color_list = ['r','b']

for no, fidelity_eng in enumerate(fidelity_eng_list):
    # print(fidelity_eng_list)
    """
    TRUE (PER BETA)
    """
    true_one_sigma_free_en_list_per_beta = []
    true_two_sigma_free_en_list_per_beta = []
    true_one_sigma_expval_list_per_beta = []
    true_two_sigma_expval_list_per_beta = []
    true_one_sigma_entropy_list_per_beta = []
    true_two_sigma_entropy_list_per_beta = []
    true_mean_free_energy_list_per_beta = []
    true_mean_expval_list_per_beta = []
    true_mean_entropy_list_per_beta = []
    true_ground_state_per_beta = []

    
    """
    OBTAINED (PER BETA)
    """
    obtained_one_sigma_free_en_list_per_beta = []
    obtained_two_sigma_free_en_list_per_beta = []
    obtained_one_sigma_expval_list_per_beta = []
    obtained_two_sigma_expval_list_per_beta = []
    obtained_one_sigma_entropy_list_per_beta = []
    obtained_two_sigma_entropy_list_per_beta = []
    obtained_mean_free_energy_list_per_beta = []
    obtained_mean_expval_list_per_beta = []
    obtained_mean_entropy_list_per_beta = []

    for inverse_temp in inverse_temp_list:
        # print(f'{neural_net}, inverse temp:', inverse_temp)
        if inverse_temp == '5p2':
            inverse_temp_ham = 5.2

        """
        OBTAINED (PER INST)
        """
        obtained_free_energy_per_inst = []
        obtained_entropy_per_inst = []
        obtained_expval_per_inst = []

        """
        TRUE (PER INST)
        """
        true_free_energy_per_inst = []
        true_expval_per_inst = []
        true_entropy_per_inst = []
        errors_per_inst = []
        true_ground_state_per_inst = []

        twoq_gate_list_per_eng = []


        if qubits == 6:
            inst_list = [0,2,4,6]
        elif qubits == 4:
            inst_list = [0,4,6,9]
        elif qubits == 7:
            if inverse_temp == '5p2':
                inst_list = [0,2,4]
            elif inverse_temp == 18:
                inst_list = [2,4,9]
            elif inverse_temp == 35:
                inst_list = [0,2,4,9]
            else:
                inst_list = [0,2,6,4,9]
        else:
            inst_list = [0,2,4,6,9]

        # LOADING DATA FROM AGENT
        for inst in inst_list:
        # for inst in [0]:
            # TURE ENERGY CALCULATION
            file = f'SYK_{qubits}q_inst{inst}_ham'
            ham_data = np.load(f"ham_data_syk/{file}.npz")
            H = ham_data['hamiltonian']
            if inverse_temp == "5p2":
                true_free_energy, true_expval, true_entropy, true_thermal_state = helmholtz_free_energy(H, inverse_temp_ham)
            else:
                true_free_energy, true_expval, true_entropy, true_thermal_state = helmholtz_free_energy(H, inverse_temp)

            true_free_energy_per_inst.append(true_free_energy)
            true_expval_per_inst.append(true_expval)
            true_entropy_per_inst.append(true_entropy)
            true_ground_state_per_inst.append(np.min(la.eig(H)[0].real))

            """
            Loading the training data as .json object
            """
            file = f'vanilla_cobyla_SYK_{qubits}q_inst{inst}_layer{layer}_cnn_finite_beta{inverse_temp}'
            if fidelity_eng:
                with open(f'noiseless/results/3D-CNN/{file}/summary_{seed}.json', 'r') as openfile:
                    data = json.load(openfile)
            else:
                with open(f'noiseless/results/3D-CNN/{file}/summary_{seed}.json', 'r') as openfile:
                    data = json.load(openfile)

            error_list  = []
            error_list_expval  = []
            error_list_entropy  = []
            succ_ep_list = []
            state_fidelity_list = []
            time_list = []

            """
            EXTRACTING THE ERROR IN: 
                - Free energy
                - Hamiltonian expectation
                - Entropy

            FOR ALL THE EPISODES
            """
            tot_episodes = len(data['train'].keys())
            for ep in range(tot_episodes):
                err = data['train'][f'{ep}']['errors'][-1]
                err_expval = data['train'][f'{ep}']['expval'][-1]
                err_entropy = data['train'][f'{ep}']['entropy'][-1]
                error_list_expval.append(np.abs(err_expval-true_expval))
                error_list_entropy.append(np.abs(err_entropy-true_entropy))

                """
                ERROR EXTRACTION:
                 - As per the Section 4 of the preprint
                """

                if fidelity_eng:
                    if qubits == 4:
                        w = 0.5
                    elif qubits == 5:
                        w = 1.02
                    elif qubits == 6:
                        w = 2
                    elif qubits == 7:
                        w = 2
                else:
                    if qubits == 4:
                        w = 0.8
                    elif qubits == 5:
                        w = 1.01
                    elif qubits == 6:
                        w = 1.16
                    elif qubits == 7:
                        w = 2
                if qubits == 7:
                    error_list.append(err+w*np.abs(err_entropy-true_entropy))
                else:
                    error_list.append(err+w*np.abs(err_expval-true_expval))

            errors_per_inst.append(np.min(error_list))
            twoq_gate_list, oneq_gate_list, gate_num_list, depth_list, circ_list = [], [], [], [], []


            """
            CHOOSING THE BEST RL-AGENT PROPOSED CIRCUIT BY MINIMIZING THE ERROR AND MAKING THE CIRCUIT
            """
            succ_ep_list = [ np.argmin(error_list) ]
            for succ_ep in succ_ep_list:
                actions = data['train'][f'{succ_ep}']['actions']
                err = data['train'][f'{ep}']['errors'][-1]
                exp_val = data['train'][f'{succ_ep}']['expval'][-1]
                entropy = data['train'][f'{succ_ep}']['entropy'][-1]

                if inverse_temp == "5p2":
                    obtained_free_energy_per_inst.append(exp_val-(1/inverse_temp_ham)*entropy)
                else:
                    obtained_free_energy_per_inst.append(exp_val-(1/inverse_temp)*entropy)
                obtained_entropy_per_inst.append(entropy)
                obtained_expval_per_inst.append(exp_val)
                circuit = QuantumCircuit(qubits)
                for a in actions:
                    action = dictionary_of_actions(qubits)[a]
                    final_circuit = make_circuit_qiskit(action, qubits, circuit)
                gate_info = final_circuit.count_ops()
                key_list = gate_info.keys()
                one_gate, two_gate = 0,0
                for k in key_list:
                    if k == 'cx':
                        two_gate += gate_info[k]
                    else:
                        one_gate += gate_info[k]
                circ_list.append(final_circuit)
                twoq_gate_list.append(two_gate)
                oneq_gate_list.append(one_gate)
                gate_num_list.append(one_gate+two_gate)        
                depth_list.append(final_circuit.depth())
            twoq_gate_list_per_eng.append(twoq_gate_list[0])

        print(f'CX corresponds to min err ({inverse_temp}, fidelity reward {fidelity_eng})', twoq_gate_list_per_eng)

        """
        TRUE GROUND STATE
        """
        true_ground_state_per_beta.append(np.mean(true_ground_state_per_inst))

        """
        ERRORS
        """ 
        true_one_sigma_free_en_list_per_beta.append(np.std(true_free_energy_per_inst))
        true_two_sigma_free_en_list_per_beta.append(2*np.std(true_free_energy_per_inst))

        true_one_sigma_expval_list_per_beta.append(np.std(true_expval_per_inst))
        true_two_sigma_expval_list_per_beta.append(2*np.std(true_expval_per_inst))

        true_one_sigma_entropy_list_per_beta.append(np.std(true_entropy_per_inst))
        true_two_sigma_entropy_list_per_beta.append(2*np.std(true_entropy_per_inst))

        #  ---------------------------------------------- #

        obtained_one_sigma_free_en_list_per_beta.append(np.std(obtained_free_energy_per_inst))
        obtained_two_sigma_free_en_list_per_beta.append(2*np.std(obtained_free_energy_per_inst))

        obtained_one_sigma_expval_list_per_beta.append(np.std(obtained_expval_per_inst))
        obtained_two_sigma_expval_list_per_beta.append(2*np.std(obtained_expval_per_inst))

        obtained_one_sigma_entropy_list_per_beta.append(np.std(obtained_entropy_per_inst))
        obtained_two_sigma_entropy_list_per_beta.append(2*np.std(obtained_entropy_per_inst))


        """
        VARIABLES
        """
        true_mean_free_energy_list_per_beta.append(np.mean(true_free_energy_per_inst))
        true_mean_expval_list_per_beta.append(np.mean(true_expval_per_inst))
        true_mean_entropy_list_per_beta.append(np.mean(true_entropy_per_inst))

        obtained_mean_free_energy_list_per_beta.append(np.mean(obtained_free_energy_per_inst))
        obtained_mean_expval_list_per_beta.append(np.mean(obtained_expval_per_inst))
        obtained_mean_entropy_list_per_beta.append(np.mean(obtained_entropy_per_inst))



    """
    OBTAINED VALUE PLOT
    """

    inverse_temp_list_plot = []
    for temp in inverse_temp_list:
        if temp == "5p2":
            inverse_temp_list_plot.append(5.2)
        else:
            inverse_temp_list_plot.append(temp)

    ax[0].errorbar(inverse_temp_list_plot, obtained_mean_free_energy_list_per_beta, yerr=obtained_one_sigma_free_en_list_per_beta, capsize=4, fmt=f"{color_list[no]}"+f"{marker_list[no]}", ecolor = f"{color_list[no]}", fillstyle = 'none', label = label_list[no])
    ax[1].errorbar(inverse_temp_list_plot, obtained_mean_expval_list_per_beta, yerr=obtained_one_sigma_expval_list_per_beta, capsize=4, fmt=f"{color_list[no]}"+f"{marker_list[no]}", ecolor = f"{color_list[no]}", fillstyle = 'none', label = label_list[no])
    ax[2].errorbar(inverse_temp_list_plot, obtained_mean_entropy_list_per_beta, yerr=obtained_one_sigma_entropy_list_per_beta, capsize=4, fmt=f"{color_list[no]}"+f"{marker_list[no]}", ecolor = f"{color_list[no]}", fillstyle = 'none', label = label_list[no])
    
    legend_elements = [
    mlines.Line2D([], [], color=color_list[0], marker='o', fillstyle = 'none', linestyle='None', markersize=7, label=label_list[0]),
    mlines.Line2D([], [], color=color_list[1], marker='x', linestyle='None', markersize=7, label=label_list[1])
                        ]

    # Add the custom legend to the axis
    if qubits in [5,7]:
        ax[0].legend(handles=legend_elements, loc='lower right',fontsize = 14)

    ax[0].grid(color = 'k', alpha = 0.3, linestyle = '--')
    ax[1].grid(color = 'k', alpha = 0.3, linestyle ='--')
    ax[2].grid(color = 'k', alpha = 0.3, linestyle ='--')

    """
    TRUE VALUE PLOT
    """
    if (error_only or fidelity_only):
        ax[0].plot(inverse_temp_list_plot, true_mean_free_energy_list_per_beta, 'k')#, alpha = 0.4)
        ax[1].plot(inverse_temp_list_plot, true_mean_expval_list_per_beta, 'k')#,alpha = 0.4)
        ax[2].plot(inverse_temp_list_plot, true_mean_entropy_list_per_beta, 'k')#,alpha = 0.4)
    else:
        if fidelity_eng == 0:
            ax[0].plot(inverse_temp_list_plot, true_mean_free_energy_list_per_beta, 'k')#, alpha = 0.4)
            ax[1].plot(inverse_temp_list_plot, true_mean_expval_list_per_beta, 'k')#,alpha = 0.4)
            ax[2].plot(inverse_temp_list_plot, true_mean_entropy_list_per_beta, 'k')#,alpha = 0.4)


    # Fill between for ±1σ
    """
    Free energy
    """
    one_sigma_free_en_lower_fill = []
    one_sigma_free_en_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        one_sigma_free_en_lower_fill.append(true_mean_free_energy_list_per_beta[i] - true_one_sigma_free_en_list_per_beta[i])
        one_sigma_free_en_upper_fill.append(true_mean_free_energy_list_per_beta[i] + true_one_sigma_free_en_list_per_beta[i])
    if (error_only or fidelity_only):
        ax[0].fill_between(inverse_temp_list_plot, one_sigma_free_en_lower_fill, one_sigma_free_en_upper_fill, color="green", alpha=0.3, label="±1σ")
    else:
        if fidelity_eng == 0:
            ax[0].fill_between(inverse_temp_list_plot, one_sigma_free_en_lower_fill, one_sigma_free_en_upper_fill, color="green", alpha=0.3, label="±1σ")


    """
    Expval
    """
    one_sigma_expval_lower_fill = []
    one_sigma_expval_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        one_sigma_expval_lower_fill.append(true_mean_expval_list_per_beta[i] - true_one_sigma_expval_list_per_beta[i])
        one_sigma_expval_upper_fill.append(true_mean_expval_list_per_beta[i] + true_one_sigma_expval_list_per_beta[i])
    if (error_only or fidelity_only):
        ax[1].fill_between(inverse_temp_list_plot, one_sigma_expval_lower_fill, one_sigma_expval_upper_fill, color="green", alpha=0.3, label="±1σ")
    else:
        if fidelity_eng == 0:
            ax[1].fill_between(inverse_temp_list_plot, one_sigma_expval_lower_fill, one_sigma_expval_upper_fill, color="green", alpha=0.3, label="±1σ")

    """
    Entropy
    """
    one_sigma_entropy_lower_fill = []
    one_sigma_entropy_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        one_sigma_entropy_lower_fill.append(true_mean_entropy_list_per_beta[i] - true_one_sigma_entropy_list_per_beta[i])
        one_sigma_entropy_upper_fill.append(true_mean_entropy_list_per_beta[i] + true_one_sigma_entropy_list_per_beta[i])
    
    if (error_only or fidelity_only):
        ax[2].fill_between(inverse_temp_list_plot, one_sigma_entropy_lower_fill, one_sigma_entropy_upper_fill, color="green", alpha=0.3, label="±1σ")
    else:
        if fidelity_eng == 0:
            ax[2].fill_between(inverse_temp_list_plot, one_sigma_entropy_lower_fill, one_sigma_entropy_upper_fill, color="green", alpha=0.3, label="±1σ")


    # Fill between for ±2σ
    """
    Free energy
    """
    two_sigma_free_en_lower_fill = []
    two_sigma_free_en_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        two_sigma_free_en_lower_fill.append(true_mean_free_energy_list_per_beta[i] - true_two_sigma_free_en_list_per_beta[i])
        two_sigma_free_en_upper_fill.append(true_mean_free_energy_list_per_beta[i] + true_two_sigma_free_en_list_per_beta[i])
    
    if (error_only or fidelity_only):
        ax[0].fill_between(inverse_temp_list_plot, two_sigma_free_en_lower_fill, two_sigma_free_en_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    else:
        if fidelity_eng == 0:
            ax[0].fill_between(inverse_temp_list_plot, two_sigma_free_en_lower_fill, two_sigma_free_en_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    """
    Expval
    """
    two_sigma_expval_lower_fill = []
    two_sigma_expval_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        two_sigma_expval_lower_fill.append(true_mean_expval_list_per_beta[i] - true_two_sigma_expval_list_per_beta[i])
        two_sigma_expval_upper_fill.append(true_mean_expval_list_per_beta[i] + true_two_sigma_expval_list_per_beta[i])

    if (error_only or fidelity_only):
        ax[1].fill_between(inverse_temp_list_plot, two_sigma_expval_lower_fill, two_sigma_expval_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    else:
        if fidelity_eng == 0:
            ax[1].fill_between(inverse_temp_list_plot, two_sigma_expval_lower_fill, two_sigma_expval_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    """
    Entropy
    """
    two_sigma_entropy_lower_fill = []
    two_sigma_entropy_upper_fill = []
    for i in range(len(inverse_temp_list_plot)):
        two_sigma_entropy_lower_fill.append(true_mean_entropy_list_per_beta[i] - true_two_sigma_entropy_list_per_beta[i])
        two_sigma_entropy_upper_fill.append(true_mean_entropy_list_per_beta[i] + true_two_sigma_entropy_list_per_beta[i])

    if (error_only or fidelity_only):
        ax[2].fill_between(inverse_temp_list_plot, two_sigma_entropy_lower_fill, two_sigma_entropy_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    else:
        if fidelity_eng == 0:
            ax[2].fill_between(inverse_temp_list_plot, two_sigma_entropy_lower_fill, two_sigma_entropy_upper_fill, color="yellow", alpha=0.2, label="±2σ")
    """
    TRUE GROUND STATE PLOT
    """
    ax[0].set_ylabel('Free energy')
    ax[1].set_ylabel('$\\langle H\\rangle_\\beta$')
    ax[2].set_ylabel('Entropy')

    ax[2].set_xlabel('$\\beta$')
    ax[2].set_xticks(inverse_temp_list_plot)

    plt.tight_layout()

ax[1].plot(inverse_temp_list_plot, [np.mean(true_ground_state_per_beta)]*len(inverse_temp_list_plot), 'k--')
plt.show()