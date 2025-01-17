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

**For an example we improve CNOT count by at least 100-fold (for N > 10) compared to first-order Trotterization!**
<p align="center">
  <img src="pics/cnot_count.png" alt="RLVTSP" width="400" height="auto">
</p>


## Why CNN?

A binary encoding scheme encodes quantum circuit into a 3D tensor as RL-state capturing the order and arrangement of gates. CNN instead of a FNN is prefered because of the following reasons:

    1. Unlike methods that flatten the 3D encoding into a single dimension (e.g. in FNNs) this scheme retains the full 3D structure.

    2. The spatial structure is directly leveraged using a 3D-CNN, enabling more effective learning and representation of quantum circuit features.

This encoding scheme improves the agent's ability to process and analyze quantum circuits as can be seen here:

![RLVTSP](pics/cnn_vs_fnn.png)

