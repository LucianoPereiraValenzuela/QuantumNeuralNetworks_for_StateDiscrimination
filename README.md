# Quantum Neural Networks for State Discrimination

## Abstract
The quantum state discrimination is an important task in many quantum technologies
However, it is not easy to find the optimal measurement strategy to discriminate non-orthogonal states.
Quantum neural networks are an efficient alternative to solve this problem. 

In this project, we implement a Qiskit library to train quantum neural networks to find the optimal generalized measurement that discriminates non-orthogonal quantum states. 

How it works? The system has three steps:

1. Generalized Measurements: Build circuits that implement generalized measurements [1].
2. Neural Networks: Use generalized measurement to build quantum neural networks [2].
3. Training: Train the neural network to find the optimal discriminator between non-orthogonal quantum states.

[1] Y. S. Yordanov and C. H. W. Barnes, "Implementation of a general single-qubit positive operator-valued 
measure on a circuit-based quantum computer"

[2] A. Patterson et al., "Quantum state discrimination using noisy quantum neural networks"

## Members
- Luciano Pereira Valenzuela
- Rafael González López
- Miguel Ángel Palomo Marcos
- Alejandro Bravo
- Rubén Romero García

## Deliverable
Github repository with the code and notebooks with examples of the usage. 

### Tests and examples:

- [Test Quantum States](qnn/tests/test_quantum_states.ipynb): Tests the generation of both defined and random quantum states using the QuantumState class.
- [Test Objective Function](qnn/tests/test_objective_function.ipynb): Tests the outcome of the cost function for two and three random quantum states both with and without noise.
- [Test POVM Circuit](qnn/tests/test_povm_circuit.ipynb): Tests the POVM circuits for two, three and four elements.
- [Test Parameter Decomposition](qnn/tests/test_parameter_decompose.ipynb): Tests the decomposition of the parameters of the POVM circuits for two and three elements.
- [Test Discrimination with Inconclusive Outcome](qnn/tests/test_discrimination_with_inconclusive_outcome.ipynb): Tests the evolution of the error probability when the inconclusiveness probability is added in the cost function.
- [Test Minimum Error Discrimination](qnn/tests/test_minimum_error_discrimination.ipynb): Tests the evolution of the error probability with the number of evaluations.
- [Test Optimizers Performance](qnn/tests/test_optimizer.ipynb): Tests how different optimizers such as SPSA, ADAM or COBYLA behave when doing state discrimination.

### Full experiments
 
- [Display of the POVM in the Bloch Sphere](qnn/results/Obtain_and_plot_povm.ipynb): Presents the quantum states to discriminate and the evolution of the optimal POVM we propose in the Bloch sphere.

https://user-images.githubusercontent.com/31738826/142266521-7c95252d-3eb8-493f-8282-c36ad0e85d7d.mp4

- [Comparison with the Helstrom Bound](qnn/results/Comparison_HelstromBound.ipynb): Compares our results with the Helstrom Bound, which is the theoretical minimum-error probability there is when discriminating two non-orthogonal quantum states.
- [Experiment of our program in a real quantum computer](qnn/results/experiment_minimum_error.ipynb): We run the training of a minimum-error discriminator on the IBM quantum device "ibmq_guadalupe".

Tests

```
cd optimal_quantum_control
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pytest --cov
```
