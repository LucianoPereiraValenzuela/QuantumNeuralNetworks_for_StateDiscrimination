from unittest.mock import Mock
from qiskit import QuantumCircuit
import numpy as np

from ..quantum_state import QuantumState
from ..config import test_config as config


class TestQuantumState:

    def instance_test(self):
        states = [0, 0]
        probabilities = [1 / len(states)] * len(states)

        qs = QuantumState(states)

        assert qs._states is states
        assert qs._probabilities is probabilities

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
