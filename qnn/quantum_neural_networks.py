import logging
from qiskit import QuantumCircuit, Aer
from qiskit import transpile
from config import config


class StateDiscriminativeQuantumNeuralNetworks:
    def __init__(self, ψ, ϕ, backend='aer_simulator', shots=2 ** 10) -> None:
        # Config
        self._config = config
        self._logger = logging.getLogger(self._config.LOG_CONFIG['name'])
        self._logger.setLevel(self._config.LOG_CONFIG['level'])
        log_handler = self._config.LOG_CONFIG['stream_handler']
        log_handler.setFormatter(logging.Formatter(self._config.LOG_CONFIG['format']))
        self._logger.addHandler(log_handler)

        # Parameters
        self.ψ = ψ
        self.ϕ = ϕ
        self.backend = backend
        self.shots = shots

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
            povm.compose(gate_r1, list(range(i + 1)), inplace=True)
            povm.x(0)
            povm.compose(gate_r2, list(range(i + 1)), inplace=True)

            v1 = QuantumCircuit(1, name=f'V1({str(i)})')
            v1.u(th_v1[i - 1], fi_v1[i - 1], lam_v1[i - 1], 0)
            gate_v1 = v1.to_gate().control(i)

            v2 = QuantumCircuit(1, name=f'V2({str(i)})')
            v2.u(th_v2[i - 1], fi_v2[i - 1], lam_v2[i - 1], 0)
            gate_v2 = v2.to_gate().control(i)

            povm.x(i)
            povm.compose(gate_v1, list(range(1, i + 1)) + [0], inplace=True)
            povm.x(i)
            povm.compose(gate_v2, list(range(1, i + 1)) + [0], inplace=True)

        return povm

    def cost_function(self, params):
        # Decompose params
        n = len(params) // 11
        th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2 = [params[i * n:(i + 1) * n] for i in
                                                                                   range(len(params) // n)]

        # Create the first circuit using get_n_element_povm
        circuit = self.get_n_element_povm(n+1, th_u, fi_u, lam_u, th1, th2, th_v1, th_v2, fi_v1, fi_v2, lam_v1, lam_v2)

        # Create the psi circuit
        qc_ψ = QuantumCircuit(2, 1)
        qc_ψ.initialize(self.ψ, 0)
        qc_ψ.barrier()
        qc_ψ.compose(circuit, [0, 1], inplace=True)
        qc_ψ.measure(1, 0)

        # Create the phi circuit
        qc_ϕ = QuantumCircuit(2, 1)
        qc_ϕ.initialize(self.ϕ, 0)
        qc_ϕ.barrier()
        qc_ϕ.compose(circuit, [0, 1], inplace=True)
        qc_ϕ.measure(1, 0)

        # Create the backend
        backend_sim = Aer.get_backend(self.backend)

        # Transpile and run
        qc_ψ = transpile(qc_ψ, backend_sim)
        results_ψ = backend_sim.run(qc_ψ, self.shots)
        qc_ϕ = transpile(qc_ψ, backend_sim)
        results_ϕ = backend_sim.run(qc_ϕ, self.shots)

        # Count
        counts_ψ = results_ψ.result().get_counts()
        counts_ϕ = results_ϕ.result().get_counts()

        # Get prob
        p_1_ψ = counts_ψ.get('1', 0) / self.shots
        p_0_ϕ = counts_ϕ.get('0', 0) / self.shots
        # p_1_ϕ = counts_ϕ.get('1', 0) / shots
        # p_0_ψ = counts_ψ.get('0', 0) / shots
                
        return 0.5 * p_1_ψ + 0.5 * p_0_ϕ
    
    
    
    
    
    
