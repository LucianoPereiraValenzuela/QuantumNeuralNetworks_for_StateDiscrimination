from unittest.mock import Mock
from qiskit import QuantumCircuit
import numpy as np

from ..quantum_neural_networks import StateDiscriminativeQuantumNeuralNetworks


class TestStateDiscriminativeQuantumNeuralNetworks:

    def instance_test(self):
        psi = [0]
        phi = [0]
        alpha_1 = 0.1
        alpha_2 = 0.1

        qnn = StateDiscriminativeQuantumNeuralNetworks(psi, phi, alpha_1, alpha_2)

        assert qnn._psi is psi
        assert qnn._phi is phi
        assert qnn._alpha_1 is alpha_1
        assert qnn._alpha_2 is alpha_2

    def cost_function_test(self):
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
