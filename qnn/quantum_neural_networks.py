import logging
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers import Backend
from qiskit.algorithms.optimizers import Optimizer
from config import config
from typing import Optional


class StateDiscriminativeQuantumNeuralNetworks:
    def __init__(
            self,
            states: [np.array],
            alpha_1: float,
            alpha_2: float,
            backend: Backend = Aer.get_backend('aer_simulator'),
            shots: int = 2 ** 10) -> None:
        """Constructor.
        Includes the config and logger as well as the main params.

        Parameters
        -------
        states
            Array of quantum states
        backend
            Qiskit backend
        shots
            Number of simulations
        """

        self._config = config
        self._logger = logging.getLogger(self._config.LOG_CONFIG['name'])
        self._logger.setLevel(self._config.LOG_CONFIG['level'])
        log_handler = self._config.LOG_CONFIG['stream_handler']
        log_handler.setFormatter(logging.Formatter(self._config.LOG_CONFIG['format']))
        self._logger.addHandler(log_handler)

        self._states = states
        self._backend = backend
        self._shots = shots
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2

    def cost_function(self, params) -> float:
        """Cost function.

        Parameters
        -------
        params
            A flat list of all the parameters.

        Returns
        -------
        The cost.
        """

        p = self.decompose_parameters(params)
        if not p:
            self._logger.error('Cannot calculate the cost function with these parameters.')
            return 0

        # Create the first circuit using get_n_element_povm
        circuit = self.get_n_element_povm(
            p['n'], p['theta_u'], p['phi_u'], p['lambda_u'], p['theta_1'], p['theta_2'], p['theta_v1'],
            p['theta_v2'], p['phi_v1'], p['phi_v2'], p['lambda_v1'], p['lambda_v2'])

        measurements = []
        for state in self._states:
            # Create the psi circuit
            qc = QuantumCircuit(p['n'], p['n'] - 1)
            qc.initialize(state, 0)
            qc.barrier()
            qc.compose(circuit, list(range(p['n'])), inplace=True)
            qc.measure(range(1, p['n']), range(p['n'] - 1))
            measurements.append(qc)

        # Transpile and run
        qc = transpile(measurements, self._backend)
        jobs = self._backend.run(qc, shots=self._shots)
        results = jobs.result().get_counts()

        if self._alpha_2 == 0:
            if p['n'] != np.ceil(np.log2(len(self._states))) + 1:
                raise Exception("Inconsistent amount of outcomes")
            prob = 0
            for i in range(len(self._states)):
                prob += 1 - results[i].get(bin(i)[2:].zfill(p['n'] - 1), 0) / self._shots
                if (i == len(self._states) - 1) and (len(self._states) % 2 == 1):
                    prob -= results[i].get(bin(i + 1)[2:].zfill(p['n'] - 1), 0) / self._shots
            return prob / len(self._states)
        else:
            if p['n'] != np.ceil(np.log2(len(self._states) + 1)) + 1:
                raise Exception("Inconsistent amount of outcomes")
            prob_error = 0
            prob_inc = 0
            for i in range(len(self._states)):
                prob_error += 1 - results[i].get(bin(i)[2:].zfill(p['n'] - 1), 0) / self._shots
                prob_inc += results[i].get(bin(len(self._states) - 1)[2:].zfill(p['n'] - 1), 0) / self._shots
                if len(self._states) % 2 == 0:
                    prob_inc += results[i].get(bin(len(self._states))[2:].zfill(p['n'] - 1), 0) / self._shots

            return self._alpha_1 * prob_error / len(self._states) + self._alpha_2 * prob_inc / len(self._states)

    def decompose_parameters(self, parameters: list) -> Optional[dict]:
        """Qiskit optimizations require a 1-dimension array, thus the
        params should be passed as a list. However, that makes the code
        very difficult to understand - that's why internally the params
        are decomposed.

        Parameters
        -------
        parameters
            List with all the required parameters

        Returns
        -------
        A dictionary with the parameters or None
        """

        if not ((len(parameters) - 3) % 8 == 0):
            self._logger.error('Parameter list length is not consistent. Should be groups of 11 items.')
            return None

        n = (len(parameters) - 3) // 8
        u_params = [[parameters[0]], [parameters[1]], [parameters[2]]]
        parameters = parameters[3:]
        param_list = [parameters[i * n:(i + 1) * n] for i in range(len(parameters) // n)]

        return {
            'n': n + 1,
            'theta_u': u_params[0],
            'phi_u': u_params[1],
            'lambda_u': u_params[2],
            'theta_1': param_list[0],
            'theta_2': param_list[1],
            'theta_v1': param_list[2],
            'theta_v2': param_list[3],
            'phi_v1': param_list[4],
            'phi_v2': param_list[5],
            'lambda_v1': param_list[6],
            'lambda_v2': param_list[7],
        }

    def discriminate(self, optimizer: Optimizer, initial_params: [float]):
        """Performs optimization using the given optimizer and a flat
        list of parameters. Uses the cost function defined above.

        Parameters
        -------
        optimizer
            Optimizer method from Qiskit.algorithm.optimizers.
        initial_params
            Flat list of parameters.

        Returns
        -------
        Result of the optimization.
        """
        return optimizer.optimize(len(initial_params), self.cost_function, initial_point=initial_params)

    @staticmethod
    def get_n_element_povm(
            n: int, theta_u: [float], phi_u: [float], lambda_u: [float], theta1: [float], theta2: [float],
            theta_v1: [float], theta_v2: [float], phi_v1: [float], phi_v2: [float], lambda_v1: [float],
            lambda_v2: [float]
    ) -> QuantumCircuit:
        """Creates the n-element POVM, using the method proposed in
        'Implementation of a general single-qubit positive operator-valued
        measure on a circuit-based quantum computer' by Yordanov and Barnes.

        Parameters
        -------
        n
            Number of modules in the POVM
        theta_u
            TBD
        phi_u
            TBD
        lambda_u
            TBD
        theta1
            TBD
        theta2
            TBD
        theta_v1
            TBD
        theta_v2
            TBD
        phi_v1
            TBD
        phi_v2
            TBD
        lambda_v1
            TBD
        lambda_v2
            TBD

        Returns
        -------
        The circuit of an n-element POVM with the given parameters.
        """

        povm = QuantumCircuit(n, name='POVM_n')
        povm.u(theta_u[0], phi_u[0], lambda_u[0], 0)

        for i in range(1, n):
            r1 = QuantumCircuit(1, name=f'R1({str(i)})')
            r1.ry(theta1[i - 1], 0)
            gate_r1 = r1.to_gate().control(i)

            r2 = QuantumCircuit(1, name=f'R2({str(i)})')
            r2.ry(theta2[i - 1], 0)
            gate_r2 = r2.to_gate().control(i)

            povm.x(0)
            povm.compose(gate_r1, list(range(i + 1)), inplace=True)
            povm.x(0)
            povm.compose(gate_r2, list(range(i + 1)), inplace=True)

            v1 = QuantumCircuit(1, name=f'V1({str(i)})')
            v1.u(theta_v1[i - 1], phi_v1[i - 1], lambda_v1[i - 1], 0)
            gate_v1 = v1.to_gate().control(i)

            v2 = QuantumCircuit(1, name=f'V2({str(i)})')
            v2.u(theta_v2[i - 1], phi_v2[i - 1], lambda_v2[i - 1], 0)
            gate_v2 = v2.to_gate().control(i)

            povm.x(i)
            povm.compose(gate_v1, list(range(1, i + 1)) + [0], inplace=True)
            povm.x(i)
            povm.compose(gate_v2, list(range(1, i + 1)) + [0], inplace=True)

        return povm


def helstrom_bound(psi: np.array, phi: np.array) -> float:
    """Calculates the Helstrom bound, optimal error.

    Parameters
    -------
    psi
        First quantum state.
    phi
        Second quantum state

    Returns
    -------
    Helstrom bound
    """
    return 0.5 - 0.5 * np.sqrt(1 - abs(np.vdot(psi, phi)) ** 2)


def random_quantum_state():
    """Creates a random quantum state.

    Returns
    -------
    A quantum state
    """
    z0 = np.random.randn(2) + 1j * np.random.randn(2)
    z0 = z0 / np.linalg.norm(z0)
    return z0
