from unittest.mock import patch

import numpy as np

from qnn.quantum_state import QuantumState


class TestQuantumState:

    def instance_test(self):
        states = [0, 0]
        probabilities = [1 / len(states)] * len(states)

        qs = QuantumState(states)

        assert qs._states is states
        assert np.testing.assert_array_equal(qs._probabilities, probabilities) is None

    def probabilities_test(self):
        states = [0, 0]
        probabilities = [1.0, 1.0]
        qs = QuantumState(states, probabilities)

        result = qs.probabilities
        assert result == probabilities

    def states_test(self):
        states = [0, 0]
        probabilities = [1.0, 1.0]
        qs = QuantumState(states, probabilities)

        result = qs.states
        assert result == states

    def random_ok_test(self):
        states = [0, 0]
        probabilities = [1.0, 1.0]
        qs = QuantumState(states, probabilities)

        expected_result = QuantumState(states=[np.array([0.5 + 0.5j, 0.5 + 0.5j])])

        with patch.object(QuantumState, "normalized_random_array", return_value=np.array([0.5 + 0.5j, 0.5 + 0.5j])):
            result = qs.random()

        assert np.testing.assert_array_equal(result.states, expected_result.states) is None
        assert np.testing.assert_array_equal(result.probabilities, expected_result.probabilities) is None

    def random_ok_with_invalid_input_test(self):
        states = [0, 0]
        probabilities = [1.0, 1.0]
        qs = QuantumState(states, probabilities)

        expected_result = QuantumState(states=[np.array([0.5 + 0.5j, 0.5 + 0.5j])])

        with patch.object(QuantumState, "normalized_random_array", return_value=np.array([0.5 + 0.5j, 0.5 + 0.5j])):
            result = qs.random("invalid input")

        assert np.testing.assert_array_equal(result.states, expected_result.states) is None
        assert np.testing.assert_array_equal(result.probabilities, expected_result.probabilities) is None

    def normalized_random_array_test(self):
        states = [0, 0]
        probabilities = [1.0, 1.0]
        qs = QuantumState(states, probabilities)

        expected_result = np.array([0.5+0.5j, 0.5+0.5j])

        with patch.object(np.random, "randn", return_value=np.array([1, 1])):
            result = qs.normalized_random_array()

        assert np.testing.assert_array_equal(result, expected_result) is None
