import logging
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
from config import config
from typing import Optional


class StateDiscriminativeQuantumNeuralNetworks:
    def __init__(
            self,
            psi: np.array,
            phi: np.array,
            backend: str = 'aer_simulator',
            shots: int = 2 ** 10) -> None:
        """Constructor.
        Includes the config and logger as well as the main params.

        Parameters
        -------
        psi
            First quantum state
        phi
            Second quantum state
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

        self._psi = psi
        self._phi = phi
        self._backend = backend
        self._shots = shots

    def get_n_element_povm(
            self,
            n: int,
            theta_u: [float],
            phi_u: [float],
            lambda_u: [float],
            theta1: [float],
            theta2: [float],
            theta_v1: [float],
            theta_v2: [float],
            phi_v1: [float],
            phi_v2: [float],
            lambda_v1: [float],
            lambda_v2: [float],
    ) -> QuantumCircuit:
        """Constructor.
        Includes the config and logger as well as the main params.

        Parameters
        -------
        n
            Number of elements in the circuit
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
            p['n'] + 1, p['theta_u'], p['phi_u'], p['lambda_u'], p['theta_1'], p['theta_2'], p['theta_v1'],
            p['theta_v2'], p['phi_v1'], p['phi_v2'], p['lambda_v1'], p['lambda_v2'])

        # Create the psi circuit
        qc_psi = QuantumCircuit(2, 1)
        qc_psi.initialize(self._psi, 0)
        qc_psi.barrier()
        qc_psi.compose(circuit, [0, 1], inplace=True)
        qc_psi.measure(1, 0)

        # Create the phi circuit
        qc_phi = QuantumCircuit(2, 1)
        qc_phi.initialize(self._phi, 0)
        qc_phi.barrier()
        qc_phi.compose(circuit, [0, 1], inplace=True)
        qc_phi.measure(1, 0)

        # Create the backend
        backend_sim = Aer.get_backend(self._backend)

        # Transpile and run
        qc_psi = transpile(qc_psi, backend_sim)
        results_psi = backend_sim.run(qc_psi, self._shots)
        qc_phi = transpile(qc_phi, backend_sim)
        results_phi = backend_sim.run(qc_phi, self._shots)

        # Count
        counts_psi = results_psi.result().get_counts()
        counts_phi = results_phi.result().get_counts()

        # Get prob
        p_1_psi = counts_psi.get('1', 0) / self._shots
        p_0_phi = counts_phi.get('0', 0) / self._shots
        # p_1_phi = counts_phi.get('1', 0) / shots
        # p_0_psi = counts_psi.get('0', 0) / shots

        return 0.5 * p_1_psi + 0.5 * p_0_phi

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

        if not len(parameters) % 11 == 0:
            self._logger.error('Parameter list length is not consistent. Should be groups of 11 items.')
            return None

        n = len(parameters) // 11
        param_list = [parameters[i * n:(i + 1) * n] for i in range(len(parameters) // n)]

        return {
            'n': n,
            'theta_u': param_list[0],
            'phi_u': param_list[1],
            'lambda_u': param_list[2],
            'theta_1': param_list[3],
            'theta_2': param_list[4],
            'theta_v1': param_list[5],
            'theta_v2': param_list[6],
            'phi_v1': param_list[7],
            'phi_v2': param_list[8],
            'lambda_v1': param_list[9],
            'lambda_v2': param_list[10],
        }

    def discriminate(self, optimizer, initial_params):
        return optimizer.optimize(len(initial_params),
                                  self.cost_function,
                                  initial_point=initial_params)


def HelstromBound( psi, phi ):
    return 0.5 - 0.5*np.sqrt( 1 - abs(np.vdot( psi, phi ))**2 )

def random_quantum_state():
    z0 = np.random.randn(2) + 1j * np.random.randn(2)
    z0 = z0 / np.linalg.norm(z0)
    return z0

