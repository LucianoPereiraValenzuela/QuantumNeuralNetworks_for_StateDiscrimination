import logging
from typing import Callable
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.algorithms.optimizers import Optimizer
from qiskit.providers import Backend

from qnn.config import config
from qnn.quantum_state import QuantumState

from qutip import Bloch


class StateDiscriminativeQuantumNeuralNetworks:
    def __init__(
            self,
            states: [QuantumState],
            inc_outcome: bool = False,
            alpha_1: float = 1.,
            alpha_2: float = 0.,
            backend: Backend = Aer.get_backend('aer_simulator'),
            shots: int = 2 ** 10) -> None:
        """Constructor.
        Includes the config and logger as well as the main params.

        Parameters
        -------
        states
            Array of quantum states
        inc_outcome
            Bool to include an inconclusive outcome
        alpha_1
            Weight of the error probability
        alpha_2
            Weight of the inconclusive probability
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
        self._inc_outcome = inc_outcome
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2

    def cost_function(self, params, callback: Optional[Callable] = None) -> float:
        """Cost function for the discrimination. This was proposed on the article
        "Quantum state discrimination using noisy quantum neural networks" by A. Patterson, et al.
        
        https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.013063

        Parameters
        -------
        params
            A flat list of all the parameters.
        callback
            A function to access to the intermediate data during the optimization.
            The function have to have as inputs the parameters, the qiskit results,
            the error probability, the inconclusive probability, and the objective function.
        Returns
        -------
        The cost.
        """

        p = self.decompose_parameters(params)
        if not p:
            raise Exception('Cannot calculate the cost function with these parameters.')

        # Create the first circuit using get_n_element_povm
        circuit = self.get_n_element_povm(
            p['n'], p['theta_u'], p['phi_u'], p['lambda_u'], p['theta_1'], p['theta_2'], p['theta_v1'],
            p['theta_v2'], p['phi_v1'], p['phi_v2'], p['lambda_v1'], p['lambda_v2'])

        # Create circuit
        circuit_measurements = []
        label = []
        n_noisy = []
        for i in range(len(self._states)):
            for single_state in self._states[i].states:
                qc = QuantumCircuit(p['n'], p['n'] - 1)
                qc.initialize(single_state, 0)
                qc.barrier()
                qc.compose(circuit, list(range(p['n'])), inplace=True)
                qc.measure(range(1, p['n']), range(p['n'] - 1))
                circuit_measurements.append(qc)
                label.append(bin(i)[2:].zfill(p['n'] - 1))
                n_noisy.append(len(self._states[i].states))

        # Transpile and run
        circuit_measurements = transpile(circuit_measurements, self._backend, optimization_level=3)
        jobs = self._backend.run(circuit_measurements, shots=self._shots)
        results = jobs.result().get_counts()

        if self._inc_outcome is False:
            if p['n'] != np.ceil(np.log2(len(self._states))) + 1:
                raise Exception("Inconsistent amount of outcomes")

            prob_error = 0
            for i in range(len(circuit_measurements)):
                prob_error += (1 - results[i].get(label[i], 0) / self._shots) / n_noisy[i]
                if label[i] == label[-1] and (len(self._states) % 2 == 1):
                    prob_error -= results[i].get('1' * (p['n'] - 1), 0) / (n_noisy[i] * self._shots)
            prob_error = prob_error / len(self._states)
            prob_inc = 0
            prob = prob_error
        else:
            if p['n'] != np.ceil(np.log2(len(self._states) + 1)) + 1:
                raise Exception("Inconsistent amount of outcomes")
            prob_error = 0
            prob_inc = 0
            for i in range(len(circuit_measurements)):
                prob_error += (1 - (results[i].get(label[i], 0)
                                    + results[i].get(bin(len(self._states))[2:].zfill(p['n'] - 1), 0)
                                    ) / self._shots) / n_noisy[i]

                prob_inc += results[i].get(bin(len(self._states))[2:].zfill(p['n'] - 1), 0
                                           ) / (n_noisy[i] * self._shots)
                if len(self._states) % 2 == 0:
                    prob_error -= results[i].get(bin(len(self._states) + 1)[2:].zfill(p['n'] - 1), 0
                                                 ) / (n_noisy[i] * self._shots)
                    prob_inc += results[i].get(bin(len(self._states) + 1)[2:].zfill(p['n'] - 1), 0
                                               ) / (n_noisy[i] * self._shots)

            prob_error = prob_error / len(self._states)
            prob_inc = prob_inc / len(self._states)
            prob = self._alpha_1 * prob_error + self._alpha_2 * prob_inc

        if callback is not None:
            callback(params, results, prob_error, prob_inc, prob)

        return prob

    def discriminate(self, optimizer: Optimizer, initial_params: [float], callback: Optional[Callable] = None):
        """Performs optimization using the given optimizer and a flat
        list of parameters. Uses the cost function defined above.

        Parameters
        -------
        optimizer
            Optimizer method from Qiskit.algorithm.optimizers.
        initial_params
            Flat list of parameters.
        callback
            A function to access to the intermediate data during the optimization.
            The function have to have as inputs the parameters, the qiskit results (dict of counts),
            the error probability, the inconclusive probability, and the objective function.

        Returns
        -------
        Result of the optimization.
        """
        return optimizer.optimize(
            len(initial_params), lambda params: self.cost_function(params, callback), initial_point=initial_params)

    @staticmethod
    def decompose_parameters(flat_params: [list, np.array]) -> Optional[dict]:
        """Qiskit optimizations require a 1-dimension array, thus the
        params should be passed as a list. However, that makes the code
        very difficult to understand - that's why internally the params
        are decomposed.
        
        Parameters
        -------
        flat_params
            List or np.ndarray with all the required parameters

        Returns
        -------
        A dictionary with the parameters or None
        """
        if not isinstance(flat_params, list) and not isinstance(flat_params, np.ndarray):
            raise Exception('Input parameter should be a list')

        if not (len(flat_params) - 3) % 8 == 0:
            raise Exception('Input length is not consistent. Should be three elements plus n groups of eight items.')

        # First three params belong to the U gate
        u_params = [[flat_params[0]], [flat_params[1]], [flat_params[2]]]

        # Rest of the list represent the multiple theta1, theta2, V1 and V2 gates
        parameters = flat_params[3:]
        n = len(parameters) // 8
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

    @staticmethod
    def get_n_element_povm(
            n: int, theta_u: [float], phi_u: [float], lambda_u: [float], theta1: [float], theta2: [float],
            theta_v1: [float], theta_v2: [float], phi_v1: [float], phi_v2: [float], lambda_v1: [float],
            lambda_v2: [float]
    ) -> QuantumCircuit:
        """Creates the n-element POVM, using the method proposed in
        'Implementation of a general single-qubit positive operator-valued
        measure on a circuit-based quantum computer' by Yordanov and Barnes.
        
        https://arxiv.org/abs/2001.04749

        Parameters
        -------
        n
            Number of modules in the POVM
        theta_u
        phi_u
        lambda_u
            Angles of the U gate. There's one single U gate in the circuit.
        theta1
            Angles of the first theta gate. Array - one angle for gate,
            one gate for module. There are n theta1 gates in the circuit.
        theta2
            Angles of the second theta gate. Array - one angle for gate,
            one gate for module. There are n theta2 gates in the circuit.
        theta_v1
        phi_v1
        lambda_v1
            Angles of the V1 gate. There are n V1 gates in the circuit.
        theta_v2
        phi_v2
        lambda_v2
            Angles of the V2 gate. There are n V2 gates in the circuit.

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

    @staticmethod
    def povm(n: int, theta_u: [float], phi_u: [float], lambda_u: [float], theta1: [float], theta2: [float],
             theta_v1: [float], theta_v2: [float], phi_v1: [float], phi_v2: [float], lambda_v1: [float],
             lambda_v2: [float], output='povm'):

        """Construct the POVM implemented by the circuit given by get_n_element_povm()
        
        https://arxiv.org/abs/2001.04749
        
        Parameters
        -------
        n         
            Number of modules in the POVM.
        theta_u   
            Angles of the U gate. There's one single U gate in the circuit.
        phi_u     
            Angles of the U gate. There's one single U gate in the circuit.
        lambda_u   
            Angles of the U gate. There's one single U gate in the circuit.
        theta1     
            Angles of the first theta gate. Array - one angle for gate,
            one gate for module. There are n theta1 gates in the circuit.
        theta2    
            Angles of the second theta gate. Array - one angle for gate,
            one gate for module. There are n theta2 gates in the circuit.
        theta_v1  
            Angles of the V1 gate. There are n V1 gates in the circuit.
        phi_v1    
            Angles of the V1 gate. There are n V1 gates in the circuit.
        lambda_v1 
            Angles of the V1 gate. There are n V1 gates in the circuit.
        theta_v2  
            Angles of the V2 gate. There are n V2 gates in the circuit. 
        phi_v2    
            Angles of the V2 gate. There are n V2 gates in the circuit.
        lambda_v2 
            Angles of the V2 gate. There are n V2 gates in the circuit.
        output    
            type of output. Can be 'povm', 'kraus', and 'unitary'.
            
        Returns
        -------
        list with POVM elements
        
        """

        circuit = StateDiscriminativeQuantumNeuralNetworks.get_n_element_povm(n, theta_u, phi_u, lambda_u, theta1,
                                                                              theta2,
                                                                              theta_v1, theta_v2, phi_v1, phi_v2,
                                                                              lambda_v1,
                                                                              lambda_v2)
        circuit.save_unitary()

        backend = Aer.get_backend('aer_simulator_unitary')
        circuit = transpile(circuit, backend)
        results = backend.run(circuit).result()

        u_matrix = results.get_unitary()

        if output == 'unitary':
            return u_matrix
        else:
            u_matrix = u_matrix.reshape(2 ** (n - 1), 2, 2 ** (n - 1), 2)
            m_list = []
            for m in range(2 ** (n - 1)):
                if output == 'kraus':
                    m_list.append(u_matrix[m, :, 0, :])
                elif output == 'povm':
                    m_list.append(u_matrix[m, :, 0, :].T.conj() @ u_matrix[m, :, 0, :])

            return m_list

    @staticmethod
    def plot_bloch_sphere(povm: [np.array], states: [np.array, QuantumState]):
        """Plot in the Bloch sphere the povm and the states.
        
        Parameters
        -------
            povm 
                List with the POVM elements
            states 
                List with the states
        
        Returns
        -------
            qutip Bloch sphere
        
        """
        vec_list = QuantumState.get_bloch_vector(povm)
        vec_states = []
        for state in states:
            for psi in state.states:
                vec_states.append(QuantumState.get_bloch_vector(psi))

        sphere = Bloch()
        sphere.clear()
        for v in vec_states:
            sphere.add_points(v)
        for v in vec_list:
            sphere.add_vectors(v)
        return sphere

    @staticmethod
    def helstrom_bound(psi: QuantumState, phi: np.array) -> float:
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
        return 0.5 - 0.5 * np.sqrt(1 - abs(np.vdot(psi.states[0], phi.states[0])) ** 2)
