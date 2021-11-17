from typing import Optional, List

import numpy as np


class QuantumState:
    def __init__(self, states: [np.array], probabilities: Optional[List[float]] = None) -> None:
        self._probabilities = [1 / len(states)] * len(states) if probabilities is None else probabilities
        self._states = states

    @property
    def probabilities(self) -> [float]:
        return self._probabilities

    @property
    def states(self) -> [np.array]:
        return self._states

    @staticmethod
    def random(n: int = 1):
        """Creates a QuantumState object with n random quantum states.

        Parameters
        -------
        n
            Number of states

        Returns
        -------
        A QuantumState object with n random quantum states
        """
        if not isinstance(n, int):
            n = 1

        return QuantumState(states=[QuantumState.normalized_random_array() for _ in list(range(n))])

    @staticmethod
    def normalized_random_array() -> np.array:
        """Helper method. Creates a random numpy array and normalizes it.

        Returns
        -------
        Random normalized numpy array.
        """
        z0 = np.random.randn(2) + 1j * np.random.randn(2)
        return z0 / np.linalg.norm(z0)

    @staticmethod
    def get_bloch_vector(operator):
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])

        if isinstance(operator, QuantumState):
            operator = operator.states

        if isinstance(operator, list) is False:
            operator = [operator]

        vec_list = []
        for op in operator:
            if op.ndim == 1:
                op = np.outer(op, op.T.conj())

            vec_list.append([np.real(np.trace(op @ sx)),
                             np.real(np.trace(op @ sy)),
                             np.real(np.trace(op @ sz))])

        return vec_list[0] if len(vec_list) == 1 else vec_list
