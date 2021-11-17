# Quantum Neural Networks for State Discrimination

## Abstract
The discrimination of quantum states is an important task in many quantum technologies. 
However, finding the measurement strategy to optimally discriminate two non-orthogonal states is not an easy task. 
Quantum neural networks are an efficient alternative to solve this problem. 

We propose a Qiskit module that uses quantum neural networks to find the optimal generalized measurement 
that discriminates two non-orthogonal quantum states.

How it works? The system has three steps:

1. Generalized Measurements: Build circuits that implement generalized measurements.
2. Neural Networks: Use generalized measurement to build neural networks.
3. Training: Train the neural network to find the optimal discriminator between two non-orthogonal quantum states.

[1] A. Patterson et al., "Quantum state discrimination using noisy quantum neural networks".
[2] Y. S. Yordanov and C. H. W. Barnes, "Implementation of a general single-qubit positive operator-valued 
measure on a circuit-based quantum computer"

## Members
- Luciano Pereira Valenzuela
- Rafael González López
- Miguel Ángel Palomo Marcos
- Alejandro Bravo
- Rubén Romero García

## Deliverable
Github repository with the code and notebooks with examples of the usage. 

- [Test Quantum States](qnn/tests/test_quantum_states.ipynb): Tests the generation of both defined and random quantum states using the QuantumState class.

Tests

```
cd optimal_quantum_control
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pytest --cov
```
