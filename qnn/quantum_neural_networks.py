import logging

import numpy as np
from noisyopt import minimizeSPSA
from qiskit import QuantumCircuit, pulse
from qiskit.test.mock import FakeValencia
from qiskit.circuit import Gate
from qiskit import transpile, schedule as build_schedule
from scipy.linalg import expm
from scipy.optimize import minimize

from .config import config


class StateDiscriminativeQuantumNeuralNetworks:
    def __init__(
            self
    ) -> None:
        self._config = config

        self._logger = logging.getLogger(self._config.LOG_CONFIG['name'])
        self._logger.setLevel(self._config.LOG_CONFIG['level'])
        log_handler = self._config.LOG_CONFIG['stream_handler']
        log_handler.setFormatter(logging.Formatter(self._config.LOG_CONFIG['format']))
        self._logger.addHandler(log_handler)

    def get_n_element_povm(self, n, th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2):

        povm = QuantumCircuit(n, name='POVM_n')
        povm.u(th_u[0], fi_u[0], lam_u[0], 0)

        for i in range(1, n):
            r1 = QuantumCircuit(1, name=f'R1({str(i)})')
            r1.ry(th1[i - 1], 0)
            gate_r1 = r1.to_gate().control(i)

            r2 = QuantumCircuit(1, name=f'R2({str(i)})')
            r2.ry(th2[i - 1], 0)
            gate_r2 = r2.to_gate().control(i)

            povm.x(0)
            povm.append(gate_r1, range(i + 1))
            povm.x(0)
            povm.append(gate_r2, range(i + 1))

            v1 = QuantumCircuit(1, name=f'V1({str(i)})')
            v1.u(th_v1[i - 1], fi_v1[i - 1], lam_v1[i - 1], 0)
            gate_v1 = v1.to_gate().control(i)

            v2 = QuantumCircuit(1, name=f'V2({str(i)})')
            v2.u(th_v2[i - 1], fi_v2[i - 1], lam_v2[i - 1], 0)
            gate_v2 = v2.to_gate().control(i)

            povm.x(i)
            povm.append(gate_v1, list(range(1, i + 1)) + [0])
            povm.x(i)
            povm.append(gate_v2, list(range(1, i + 1)) + [0])

        return povm
