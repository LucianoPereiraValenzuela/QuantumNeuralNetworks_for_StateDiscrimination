from unittest.mock import Mock
from qiskit import QuantumCircuit
import numpy as np

from ..quantum_neural_networks import StateDiscriminativeQuantumNeuralNetworks


class TestStateDiscriminativeQuantumNeuralNetworks:

    def instance_test(self):
        """Tests the instance
        """

        qnn = StateDiscriminativeQuantumNeuralNetworks()

        assert qnn is not None

    def get_n_element_povm_test(self):
        pi = np.pi
        n = 3
        th_u = [0]
        fi_u = [0]
        lam_u = [pi / 2]
        th1 = [pi, pi / 2]
        th2 = [pi / 2, pi]
        th_v1 = [pi, pi / 2]
        th_v2 = [pi / 2, pi]
        fi_v1 = [pi, pi / 2]
        fi_v2 = [pi / 2, pi]
        lam_v1 = [pi, pi / 2]
        lam_v2 = [pi / 2, pi]

        qnn = StateDiscriminativeQuantumNeuralNetworks()

        povm_n = qnn.get_n_element_povm(n, th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2)
        povm_n.draw()

        assert 1 == 1
