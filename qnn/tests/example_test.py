import itertools
import numpy as np
from numpy import pi
from qiskit import Aer
from qiskit.compiler import transpile
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM

from ..quantum_neural_networks import StateDiscriminativeQuantumNeuralNetworks


class TestExamples:
    def luciano_test(self):
        # Create random states
        psi = StateDiscriminativeQuantumNeuralNetworks.random_quantum_state()
        phi = StateDiscriminativeQuantumNeuralNetworks.random_quantum_state()
        # ψ = np.array([1,0])
        # φ = np.array([0,1])

        # Parameters
        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0],  [0]

        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        # Initialize Discriminator
        discriminator = StateDiscriminativeQuantumNeuralNetworks(psi, phi)

        # Calculate cost function
        results = discriminator.discriminate(SPSA(100), params)
        print(results)

        # Optimal error
        print(StateDiscriminativeQuantumNeuralNetworks.helstrom_bound(psi, phi))

        assert 1 == 1
