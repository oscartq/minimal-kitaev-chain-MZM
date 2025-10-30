import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Session

from functions import gauss_state_qiskit, orbital_combinations, compute_bdg_spectrum

# def make_hardware_estimator(backend_name: str, shots=20000, resilience_level=1):
#     service = QiskitRuntimeService()
#     session = Session(service=service, backend=backend_name)
#     est = RuntimeEstimator(
#         session=session,
#         options={
#             "execution": {"shots": shots},
#             # 0: none, 1: measurement mitigation, 2–3: more advanced
#             "resilience_level": resilience_level
#         }
#     )
#     return session, est

# def make_noisy_estimator(shots=20000, seed=1):
#     # Create noise model from fake backend
#     backend = FakeBrisbane()
    
#     # Use EstimatorV2 from qiskit_aer.primitives with noise model in backend_options
#     est = BackendEstimatorV2(fake
#         options={
#             "backend_options": {
#                 "noise_model": noise_model,
#                 "basis_gates": fake.configuration().basis_gates,
#                 "coupling_map": fake.configuration().coupling_map,
#                 "seed_simulator": seed
#             },
#             "run_options": {
#                 "shots": shots
#             }
#         }
#     )
#     return est

def run_simulation(nqbit=2, eps_max=3.0, nsteps=20, t=-1.0, Delta=1.0, const=1.0):
    
    t1 = time.time()

    # Create Estimator instance (for exact statevector simulation)
    #estimator = StatevectorEstimator()
    backend = FakeBrisbane()
    estimator = BackendEstimatorV2(backend=backend)

    print(f"Starting simulation with nqbit={nqbit}, eps range: -{eps_max} to +{eps_max}, nsteps={nsteps}")
    print(f"Total number of eps points: {2*nsteps+1}")
    
    # Prepare orbital combinations 
    orb_comb = orbital_combinations(nqbit)
    orb_comb_str = [str(x) for x in orb_comb]
    
    # Create the qubit converter (Jordan-Wigner)
    qubit_converter = JordanWignerMapper()
    
    # ---------- Parity operators ----------
    parity_op = FermionicOp({"": 1.0, "+_0 -_0": -2.0}, num_spin_orbitals=nqbit)
    
    for i in range(1, nqbit):
        term = FermionicOp({"": 1.0, f"+_{i} -_{i}": -2.0}, num_spin_orbitals=nqbit)
        parity_op = parity_op @ term
    
    parity_op_JW = qubit_converter.map(parity_op)
    
    part_num_aux = FermionicOp({"+_0 -_0": 1.0}, num_spin_orbitals=nqbit)
    
    for i in range(1, int(nqbit)):
        part_num_aux += FermionicOp({f"+_{i} -_{i}": 1.0}, num_spin_orbitals=nqbit)
    
    part_num_JW = qubit_converter.map(part_num_aux)
    
    # ---------- Running the actual script ----------
    
    # Containers
    Energy_Diag = []
    EnQ = []
    correlation = []
    number_expectation = []
    parity = []
    eps_list = []

    # symmetric sweep: -eps_max ... 0 ... +eps_max in 2*nsteps+1 points
    deps = eps_max / nsteps
    total_pts = 2 * nsteps + 1

    for idx, k in enumerate(range(-nsteps, nsteps + 1), start=1):
        print(f"Executing step {idx}/{total_pts} (k={k})")
        eps = k * deps
        eps_list.append(eps)

        # This is the actual Hamiltonian in use.
        # Build Hamiltonian
        ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
        circ_aux1 = ""
        circ_aux2 = ""        
    
        for i in range(0, int(nqbit) - 1):
            ham_aux += FermionicOp({f"+_{i} -_{i+1}": -t}, num_spin_orbitals=nqbit)
            circ_aux1 += str(-t*const) + "[" + str(i) + "^ " + str(i+1) + "]+"
            ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
            circ_aux1 += str(-t*const) + "[" + str(i+1) + "^ " + str(i) + "]+"
            ham_aux += FermionicOp({f"+_{i} +_{i+1}": Delta}, num_spin_orbitals=nqbit)
            circ_aux1 += str(Delta) + "[" + str(i) + "^ " + str(i+1) + "^]+"
            ham_aux += FermionicOp({f"-_{i+1} -_{i}": Delta}, num_spin_orbitals=nqbit)
            circ_aux1 += str(np.conj(Delta)) + "[" + str(i+1) + " " + str(i) + "]+"
    
        for i in range(0, int(nqbit)):
            ham_aux += FermionicOp({f"+_{i} -_{i}": eps}, num_spin_orbitals=nqbit)
            circ_aux2 += str(eps) + "[" + str(i) + "^ " + str(i) + "]+"
        
        circ_aux = circ_aux1 + circ_aux2
        
        print(f"  Step {nsteps-k}: Jordan-Wigner transformation")
        hamilt_JW = qubit_converter.map(ham_aux)
        
        # Majorana operator i\gamma_1\gamma_N
        corr = \
            FermionicOp({f"+_0 +_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            + FermionicOp({f"-_0 +_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            - FermionicOp({f"+_0 -_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            - FermionicOp({f"-_0 -_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit)
        
        corr_JW = qubit_converter.map(corr)
        
        print(f"  eps = {eps}")
        EnQ_aux = []
        corr_aux = []
        parity_aux = []
        number_expectation_aux = []
        
        def psi(q):
            return gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0] #Wave function for orbital q for our Hamiltonian
        
        def val_en(q): #Energy
            circuit = psi(q)
            job = estimator.run([(circuit, hamilt_JW)])
            result = job.result()
            val = result[0].data.evs - eps * nqbit / 2 #Minus the onsite energy offset
            return val
    
        def val_cor(q): #Correlation
            circuit = psi(q)
            job = estimator.run([(circuit, corr_JW)])
            result = job.result()
            val = result[0].data.evs
            return val
    
        def val_num(q): #Number operator
            circuit = psi(q)
            job = estimator.run([(circuit, part_num_JW)])
            result = job.result()
            val = result[0].data.evs
            return val
    
        def val_par(q): #Parity operator
            circuit = psi(q)
            job = estimator.run([(circuit, parity_op_JW)])
            result = job.result()
            val = result[0].data.evs
            return val
        
        for q in range(0, 2**nqbit):
            EnQ_aux.append(val_en(q))
            corr_aux.append(val_cor(q))
            number_expectation_aux.append(val_num(q))
            parity_aux.append(val_par(q))
                
        EnQ.append(EnQ_aux)
        correlation.append(corr_aux)
        number_expectation.append(number_expectation_aux)
        parity.append(parity_aux)
        Energy_Diag.append(list(np.linalg.eigh(hamilt_JW.to_matrix())[0] - [eps*(nqbit/2)]*2**nqbit)) #Diagonalization of the normal Hamiltonian
    
    Energy_BdG, Fermi_En_Pl, Fermi_En_Mi = compute_bdg_spectrum(nqbit, nsteps, eps_max, t, Delta, const)
    
    t2 = time.time()
    print(f"\nSimulation completed in {t2-t1:.2f} seconds")
    
    # Return results as dictionary
    results = {
        'Energy_Diag': Energy_Diag,
        'EnQ': EnQ,
        'correlation': correlation,
        'number_expectation': number_expectation,
        'parity': parity,
        'eps_list': eps_list,
        'Energy_BdG': Energy_BdG,
        'Fermi_En_Pl': Fermi_En_Pl,
        'Fermi_En_Mi': Fermi_En_Mi,
        'orb_comb': orb_comb,
        'orb_comb_str': orb_comb_str,
        't': t,
        'nqbit': nqbit
    }
    
    return results

def plot_circuit(nqbit, eps, t, Delta, const, state):
    # Prepare orbital combinations
    orb_comb = orbital_combinations(nqbit)

    ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
    circ_aux1 = ""
    circ_aux2 = ""

    for i in range(int(nqbit) - 1):
        ham_aux += FermionicOp({f"+_{i} -_{i+1}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t*const}[{i}^ {i+1}]+"
        ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t*const}[{i+1}^ {i}]+"
        ham_aux += FermionicOp({f"+_{i} +_{i+1}": Delta}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{Delta}[{i}^ {i+1}^]+"
        ham_aux += FermionicOp({f"-_{i+1} -_{i}": Delta}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{np.conj(Delta)}[{i+1} {i}]+"

    for i in range(int(nqbit)):
        ham_aux += FermionicOp({f"+_{i} -_{i}": eps}, num_spin_orbitals=nqbit)
        circ_aux2 += f"{eps}[{i}^ {i}]+"

    circ_aux = circ_aux1 + circ_aux2

    def psi(q):
        # gauss_state_qiskit should return a QuantumCircuit; if it returns a tuple, index [0]
        return gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]

    circuit = psi(state)
    circuit.draw(output="latex", filename=f"figures/circuit_eps={eps}_state={state}.pdf")
    
    return


if __name__ == "__main__":
    # Run with default parameters
    results = run_simulation()
    
    # Optionally save results
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to 'simulation_results.pkl'")