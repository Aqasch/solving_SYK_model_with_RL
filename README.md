# Improved thermal state preparation of SYK model on QPU with RL and CNN

![RLVTSP](pics/SYK_with_RL_diagram.png)

## What is SYK model?

The Sachdev-Ye-Kitaev (SYK) model is a quantum mechanical system that has garnered significant attention in both condensed matter and high-energy physics due to its unique properties and connections to quantum gravity. The model consists of a large number (N) of interacting fermions in zero spatial dimensions and one time dimension. These fermions interact with each other through random all-to-all couplings, meaning each fermion can interact with any other fermion in the system4. This randomness in interactions is a key feature of the model. Key characteristics of the SYK model include:

1. **Exact solvability:** Despite its complexity, the model can be solved exactly in certain limits, making it a valuable tool for theoretical studies.

2. **Quantum chaos:** The model exhibits maximal quantum chaos, a property shared with black holes.

3. **Holographic duality:** The SYK model is believed to be related to quantum gravity in a negatively curved spacetime, specifically to black holes in nearly Anti-de Sitter (AdS) space.

4. **Emergent conformal symmetry:** At low temperatures, the model displays an approximate conformal symmetry, a feature important in quantum field theories.

5. **Relevance to condensed matter systems:** The model has potential applications in understanding strongly correlated materials and strange metals.

The model's unique combination of properties makes it a powerful tool for exploring fundamental questions in quantum mechanics, gravity, and the nature of spacetime. Its study has led to insights in areas ranging from quantum information to black hole physics, making it a focal point of research in theoretical physics.

## The main bottleneck we tackle

Preparing thermal states for large SYK systems (N > 12) on current quantum devices poses a significant challenge due to the increasing complexity of parameterized quantum circuits. To address this, we propose an innovative approach

    Combining reinforcement learning (RL) framework with convolutional neural networks, our method refines both the quantum circuit structure and its parameters. Hence overcome the limitations of traditional variational methods and make the preparation of thermal state beyond 12 Majorana fermions possible on near-term quantum hardware.

**For an example we improve CNOT count by at least 100-fold (for N > 10) compared to first-order Trotterization (for $\beta=$ inverse temperature $=5.2$)!**
<p align="center">
  <img src="pics/cnot_count.png" alt="RLVTSP" width="400" height="auto">
</p>


## Why CNN?

A binary encoding scheme encodes quantum circuit into a 3D tensor as RL-state capturing the order and arrangement of gates. CNN instead of a FNN is prefered because of the following reasons:

    1. Unlike methods that flatten the 3D encoding into a single dimension (e.g. in FNNs) this scheme retains the full 3D structure.

    2. The spatial structure is directly leveraged using a 3D-CNN, enabling more effective learning and representation of quantum circuit features.

To soldify the advantage with CNN we consider two distinct 4-qubit Hamiltonians (1) The $N=8$ SYK dense Hamiltonian and (2) an $\texttt{LiH}$ molecule with $3.4$ bond distance. This encoding scheme improves the agent's ability to process and analyze quantum circuits as can be seen here:

![RLVTSP](pics/cnn_vs_fnn.png)

# How to run the simulations?

Here we will elaborate how to run the code provided in the repository. Instead of giving an `.yml` file for the environment I am going to stepwisely install all the dependencies in the next few steps. Please note that the code was used on Ubuntu GNU/Linux 22.04.4 LTS (64-bit).



## Making the environment

**Step 1:** For this project, we use Anaconda which can be downloaded from https://www.anaconda.com/products/individual.


**Step 2:** 
```
conda env create -f syk_rl.yml
```

**Step 3:** 
```
conda activate syk_rl
```


## Noiseless simulations:

After activating the environment we will see how to run the noiseless simulations

**Step 1:** Go to the folder `noiseless`

**Step 2** If you want to run the code with `3D-CNN` then use
```
python main_syk_finite.py --seed 1 --config vanilla_cobyla_SYK_4q_inst0_layer5_cnn_finite_beta5p2 --experiment_name "3D-CNN/"
```

which means you are running the `main_syk_finite.py` file with the `3D-CNN` initialized with seed $1$ for configuration `vanilla_cobyla_SYK_4q_inst0_layer5_cnn_finite_beta5p2`. If you inspect the `main_syk_finite.py` file you can see that this is where the training of the neural network is happening.

**step 3:** The output after training is saved in `.json` format which is saved in the `results` folder.


## Noisy simulations:

In the case of noisy simulation

**Step 1:** Go to the folder `noisy`.

the remaining steps are similar to the *noiseless* scenario.


## Extracting the results:

To extract the results and plot the data (say for example the Figure 3 and 4) illustrated in the preprint please follow the details below.

**Step 1:** run the `plot_noiseless_with_rwd_compare.py`

```
python plot_noiseless_with_rwd_compare.py
```

The data for the plot can be found in the `put_zenodo_link` or you need to train the agent to store data.

## Running the best circuits on IBM Eagle r3 processor

