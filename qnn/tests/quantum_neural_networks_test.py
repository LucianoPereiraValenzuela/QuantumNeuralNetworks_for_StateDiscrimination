from unittest.mock import Mock
from qiskit import QuantumCircuit
import numpy as np

from ..quantum_neural_networks import StateDiscriminativeQuantumNeuralNetworks


class TestStateDiscriminativeQuantumNeuralNetworks:

    def instance_test(self):
        psi = [0]
        phi = [0]
        backend = 'aer_simulator'
        shots = 1

        qnn = StateDiscriminativeQuantumNeuralNetworks(psi, phi, backend, shots)

        assert qnn._psi is psi
        assert qnn._phi is phi
        assert qnn._backend is backend
        assert qnn._shots is shots

    def cost_function_test(self):
        psi = [0]
        phi = [0]

        qnn = StateDiscriminativeQuantumNeuralNetworks(psi, phi)

        pass

    def decompose_parameters_test(self):
        pass

    def discriminate_test(self):
        pass

    def get_n_element_povm_test(self):
        pass

    def helstrom_bound_test(self):
        pass

    def random_quantum_state_test(self):
        pass
