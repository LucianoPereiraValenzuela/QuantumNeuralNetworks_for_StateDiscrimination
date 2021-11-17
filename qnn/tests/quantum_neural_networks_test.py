import itertools
from unittest.mock import Mock

import numpy as np
import pytest

from qnn.quantum_neural_networks import StateDiscriminativeQuantumNeuralNetworks
from qnn.quantum_state import QuantumState


class TestStateDiscriminativeQuantumNeuralNetworks:

    def instance_test(self):
        states = [QuantumState([0, 0])]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        assert qnn._states is states

    def cost_function_happy_path_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [np.pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0], [0]
        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        expected_result = 0.5

        result = qnn.cost_function(params)

        assert np.testing.assert_almost_equal(result, expected_result, decimal=1, verbose=True) is None

    def cost_function_fail_no_params_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)
        qnn.decompose_parameters = Mock(return_value=None)

        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [np.pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0], [0]
        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        with pytest.raises(Exception) as e_info:
            qnn.cost_function(params)
            assert e_info == 'Cannot calculate the cost function with these parameters.'

    def cost_function_fail_inconsistent_outcomes_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [np.pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0], [0]
        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        with pytest.raises(Exception) as e_info:
            qnn.cost_function(params)
            assert e_info == 'Inconsistent amount of outcomes'

    def cost_function_fail_inconsistent_outcomes_with_boolean_true_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states, inc_outcome=True)

        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [np.pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0], [0]
        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        with pytest.raises(Exception) as e_info:
            qnn.cost_function(params)
            assert e_info == 'Inconsistent amount of outcomes'

    def decompose_parameters_happy_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        th_u, fi_u, lam_u = [0], [0], [0]
        th1, th2 = [0], [np.pi]
        th_v1, th_v2 = [0], [0]
        fi_v1, fi_v2 = [0], [0]
        lam_v1, lam_v2 = [0], [0]
        params = list(itertools.chain(th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2))

        expected_result = {
            'n': 2,
            'theta_u': [0],
            'phi_u': [0],
            'lambda_u': [0],
            'theta_1': [0],
            'theta_2': [np.pi],
            'theta_v1': [0],
            'theta_v2': [0],
            'phi_v1': [0],
            'phi_v2': [0],
            'lambda_v1': [0],
            'lambda_v2': [0]
        }

        result = qnn.decompose_parameters(params)
        assert result == expected_result

    def decompose_parameters_fail_no_list_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        params = "invalid params"
        with pytest.raises(Exception) as e_info:
            qnn.decompose_parameters(params)
            assert e_info == 'Input parameter should be a list'

    def decompose_parameters_fail_invalid_length_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)

        params = list(range(2))
        with pytest.raises(Exception) as e_info:
            qnn.decompose_parameters(params)
            assert e_info == 'Input length is not consistent. Should be three elements plus n groups of eight items.'

    def helstrom_bound_test(self):
        state = QuantumState([np.array([0.5 + 0.5j, 0.5 + 0.5j])])
        states = [state, state]

        qnn = StateDiscriminativeQuantumNeuralNetworks(states)
        expected_result = 0.5 - 0.5 * np.sqrt(1 - abs(np.vdot(state.states[0], state.states[0])) ** 2)

        result = qnn.helstrom_bound(state, state)
        assert result == expected_result
