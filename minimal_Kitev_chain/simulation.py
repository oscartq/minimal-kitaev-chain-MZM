import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Session

from functions import gauss_state_qiskit, orbital_combinations, compute_bdg_spectrum, majorana_op

def run_simulation(nqbit=2, eps_max=3.0, nsteps=20, t=-1.0, Delta=1.0):
    
    t1 = time.time()

    # Create Estimator instance (for exact statevector simulation)
    estimator = StatevectorEstimator()
    # backend = FakeBrisbane()
    # estimator = BackendEstimatorV2(backend=backend)

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
    
    polarization_op = SparsePauliOp("X").tensor(SparsePauliOp("X"))

    # ---------- Running the actual script ----------
    
    # Containers
    Energy_Diag = []
    EnQ = []
    correlation = []
    number_expectation = []
    parity = []
    polarization = []
    eps_list = []

    # symmetric sweep: -eps_max ... 0 ... +eps_max in 2*nsteps+1 points
    deps = eps_max / nsteps
    total_pts = 2 * nsteps + 1

    for idx, k in enumerate(range(-nsteps, nsteps + 1), start=1):
        print(f"Executing step {idx}/{total_pts} (k={k})")
        eps = k * deps
        eps_list.append(eps)

        # This is the actual Hamiltonian in use.
        # ---------- Two-site minimal Kitaev chain ---------- 
        ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
        circ_aux1 = ""
        circ_aux2 = ""        
    
        for i in range(0, int(nqbit) - 1):
            ham_aux += FermionicOp({f"+_{i} -_{i+1}": -t}, num_spin_orbitals=nqbit)
            circ_aux1 += str(-t) + "[" + str(i) + "^ " + str(i+1) + "]+"
            ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
            circ_aux1 += str(-t) + "[" + str(i+1) + "^ " + str(i) + "]+"
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
        polarization_aux = []
        
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
        
        def val_pol(q):
            circuit = psi(q)
            job = estimator.run([(circuit, polarization_op)])
            result = job.result()
            val = result[0].data.evs
            return val
        
        for q in range(0, 2**nqbit):
            EnQ_aux.append(val_en(q))
            corr_aux.append(val_cor(q))
            number_expectation_aux.append(val_num(q))
            parity_aux.append(val_par(q))
            polarization_aux.append(val_pol(q))
                
        EnQ.append(EnQ_aux)
        correlation.append(corr_aux)
        number_expectation.append(number_expectation_aux)
        parity.append(parity_aux)
        polarization.append(polarization_aux)
        Energy_Diag.append(list(np.linalg.eigh(hamilt_JW.to_matrix())[0] - [eps*(nqbit/2)]*2**nqbit)) #Diagonalization of the normal Hamiltonian
    
    Energy_BdG, Fermi_En_Pl, Fermi_En_Mi = compute_bdg_spectrum(nqbit, nsteps, eps_max, t, Delta)
    
    t2 = time.time()
    print(f"\nSimulation completed in {t2-t1:.2f} seconds")
    
    # Return results as dictionary
    results = {
        'Energy_Diag': Energy_Diag,
        'EnQ': EnQ,
        'correlation': correlation,
        'number_expectation': number_expectation,
        'parity': parity,
        'polarization': polarization,
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

def plot_circuit(nqbit, eps, t, Delta, state):
    # Prepare orbital combinations
    orb_comb = orbital_combinations(nqbit)

    ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
    circ_aux1 = ""
    circ_aux2 = ""

    for i in range(int(nqbit) - 1):
        ham_aux += FermionicOp({f"+_{i} -_{i+1}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t}[{i}^ {i+1}]+"
        ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t}[{i+1}^ {i}]+"
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

def compute_correlations(nqbit, eps, t=-1.0, Delta=1.0):

    print(f"Computing correlations for nqbit={nqbit}, eps={eps}")

    backend = FakeBrisbane()
    estimator_QPU = BackendEstimatorV2(backend=backend)
    estimator_exact = StatevectorEstimator()

    # Prepare orbital combinations 
    orb_comb = orbital_combinations(nqbit)
    orb_comb_str = [str(x) for x in orb_comb]

    qubit_converter = JordanWignerMapper()

    # --- build quadratic H (same as you had) ---
    ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
    circ_aux1, circ_aux2 = "", ""

    for i in range(nqbit - 1):
        ham_aux += FermionicOp({f"+_{i} -_{i+1}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t}[{i}^ {i+1}]+"
        ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t}[{i+1}^ {i}]+"
        ham_aux += FermionicOp({f"+_{i} +_{i+1}": Delta}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{Delta}[{i}^ {i+1}^]+"
        ham_aux += FermionicOp({f"-_{i+1} -_{i}": Delta}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{np.conj(Delta)}[{i+1} {i}]+"

    for i in range(nqbit):
        ham_aux += FermionicOp({f"+_{i} -_{i}": eps}, num_spin_orbitals=nqbit)
        circ_aux2 += f"{eps}[{i}^ {i}]+"

    circ_aux = circ_aux1 + circ_aux2

    # --- correlation operators: i γ_1 γ_k with 0-based indices
    # γ_1 in paper = γ_{2*0} = γ_0 here, i.e., first Majorana on site 0 with our mapping above
    gamma0 = majorana_op(0, nqbit)
    majcorr_operators = {}
    nMaj = 2 * nqbit
    for k in range(1, nMaj): #1-2, 1-3 , 1-4
        gammak = majorana_op(k, nqbit)
        majcorr_aux = 1.0j * (gamma0 @ gammak)  # this is Hermitian for k>0
        majcorr_operators[k] = qubit_converter.map(majcorr_aux)

    # --- allocate full [state, k] with k over 2*nqbit ---
    majcorrelations_QPU = np.full((2**nqbit, nMaj), np.nan, dtype=float)
    majcorrelations_exact = np.full((2**nqbit, nMaj), np.nan, dtype=float)
    majpolarization_QPU = np.full((2**nqbit, nqbit), np.nan, dtype=float)
    majpolarization_exact = np.full((2**nqbit, nqbit), np.nan, dtype=float)

    for q in range(2**nqbit):
        print(f"  Computing state {q+1}/{2**nqbit}: {orb_comb_str[q]}")
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]
        # --- Majorana correlations ---
        # --- Noise simulation ---
        for k in range(1, nMaj):  # skip k=0
            job = estimator_QPU.run([(circuit, majcorr_operators[k])])
            result = job.result()
            majcorrelations_QPU[q, k] = result[0].data.evs
            
        # --- Exact simulation ---
        for k in range(1, nMaj):  # skip k=0
            job = estimator_exact.run([(circuit, majcorr_operators[k])])
            result = job.result()
            majcorrelations_exact[q, k] = result[0].data.evs
            
    polarization_op = SparsePauliOp("X").tensor(SparsePauliOp("X"))
    for q in range(2**nqbit):
        print(f"  Computing state {q+1}/{2**nqbit}: {orb_comb_str[q]}")
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]
        # --- Majorana Polarization ---
        # --- Noise simulation ---
        for k in range(0, nqbit):  # skip k=0
            job = estimator_QPU.run([(circuit, polarization_op)])
            result = job.result()
            majpolarization_QPU[q, k] = result[0].data.evs
        # --- Exact simulation ---
        for k in range(0, nqbit):  # skip k=0
            job = estimator_exact.run([(circuit, polarization_op)])
            result = job.result()
            majpolarization_exact[q, k] = result[0].data.evs

    print("Correlation computation complete!")

    return {
        "correlations_QPU": majcorrelations_QPU,
        "correlations_exact": majcorrelations_exact,
        "polarization_QPU": majpolarization_QPU,
        "polarization_exact": majpolarization_exact,
        "orb_comb": orb_comb,
        "orb_comb_str": orb_comb_str,
        "eps": eps,
        "nqbit": nqbit,
        "t": t,
        "Delta": Delta
    }

def plot_correlation_vs_k(corr_results, state_indices=None):

    correlations_QPU = corr_results["correlations_QPU"]
    correlations_exact = corr_results["correlations_exact"]
    orb_comb = corr_results["orb_comb"]
    nqbit = corr_results["nqbit"]
    eps = corr_results["eps"]

    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"
    
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]
    state_indices = np.arange(2**nqbit)
    # restrict to k indices 1–3, but don’t exceed available Majoranas
    k_max = min(3,2 * nqbit - 1)  # highest k to show
    k_values = np.arange(2, k_max+2)  # [1, 2, 3] if available

    fig, ax = plt.subplots(figsize=(10, 7))
    NUM_COLORS = len(state_indices)
    cm = plt.get_cmap('tab10') if NUM_COLORS <= 10 else plt.get_cmap('gist_ncar')

    for idx, q in enumerate(state_indices):
        color = cm(idx / NUM_COLORS) if NUM_COLORS > 10 else cm(idx)
        # slice correlations to match k_values
        plt.plot(k_values, correlations_QPU[q, 1:k_max + 1], marker="*",
                 linestyle="None", color=color)
        plt.plot(k_values, correlations_exact[q, 1:k_max + 1], 
                 label=f"{fock_labels[q]}", linewidth=2, color=color)

    # Create proxy artists for the legend
    plt.plot([], [], color="black", linestyle="-", label="Exact")
    plt.plot([], [], color="black", marker="*", linestyle="None", label="QPU")

    plt.ylabel(r'$\langle i\,\gamma_{1}\gamma_k \rangle$', fontsize=18)
    plt.xlabel(r'$k$', fontsize=18)
    plt.title(rf'Majorana correlations at $\varepsilon={eps:.2f}$', fontsize=16)
    ax.set_xticks(k_values)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=10, ncol=3)
    plt.grid(True, alpha=0.3)

    plt.savefig(f"figures/plot_correlation_eps={eps:.1f}.png")
    plt.show()


def plot_polarization_vs_k(corr_results, state_indices=None):

    polarization_QPU = corr_results["polarization_QPU"]
    polarization_exact = corr_results["polarization_exact"]
    orb_comb = corr_results["orb_comb"]
    nqbit = corr_results["nqbit"]
    eps = corr_results["eps"]

    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"
    
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]
    state_indices = np.arange(2**nqbit)
    # restrict to k indices 1–3, but don’t exceed available Majoranas
    k_max = 2  # highest k to show
    k_values = np.arange(1, k_max+1)  # [1, 2, 3] if available

    fig, ax = plt.subplots(figsize=(10, 7))
    NUM_COLORS = len(state_indices)
    cm = plt.get_cmap('tab10') if NUM_COLORS <= 10 else plt.get_cmap('gist_ncar')

    for idx, q in enumerate(state_indices):
        color = cm(idx / NUM_COLORS) if NUM_COLORS > 10 else cm(idx)
        # slice correlations to match k_values
        plt.plot(k_values, polarization_QPU[q, :], marker="*",
                 linestyle="None", color=color)
        plt.plot(k_values, polarization_exact[q, :], 
                 label=f"{fock_labels[q]}", linewidth=2, color=color)

    # Create proxy artists for the legend
    plt.plot([], [], color="black", linestyle="-", label="Exact")
    plt.plot([], [], color="black", marker="*", linestyle="None", label="QPU")

    plt.ylabel(r'$\langle i\,\gamma_{1}\gamma_k \rangle$', fontsize=18)
    plt.xlabel(r'$k$', fontsize=18)
    plt.title(rf'Majorana correlations at $\varepsilon={eps:.2f}$', fontsize=16)
    ax.set_xticks(k_values)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=10, ncol=3)
    plt.grid(True, alpha=0.3)

    plt.savefig(f"figures/plot_correlation_eps={eps:.1f}.png")
    plt.show()

if __name__ == "__main__":
    # Run with default parameters
    results = run_simulation()
    
    # Optionally save results
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to 'simulation_results.pkl'")

# "cdaggercorrelations_QPU": cdaggercorrelations_QPU,
        # "cdaggercorrelations_exact": cdaggercorrelations_exact,
        # "ccorrelations_QPU": ccorrelations_QPU,
        # "ccorrelations_exact": ccorrelations_exact,

    # cdaggercorrelations_QPU = np.full((2**nqbit, nqbit), np.nan, dtype=float)
    # cdaggercorrelations_exact = np.full((2**nqbit, nqbit), np.nan, dtype=float)

    # ccorrelations_QPU = np.full((2**nqbit, nqbit), np.nan, dtype=float)
    # ccorrelations_exact = np.full((2**nqbit, nqbit), np.nan, dtype=float)

    # cdagger0 = FermionicOp({f"+_{0}": 1.0}, num_spin_orbitals=nqbit)
    # cdagger_operators = {}
    # for k in range(1, nqbit):
    #     cdaggerk = FermionicOp({f"+_{k}": 1.0}, num_spin_orbitals=nqbit)
    #     cdagger_aux = cdagger0 @ cdaggerk
    #     cdagger_operators[k] = qubit_converter.map(cdagger_aux)

    # c0 = FermionicOp({f"-_{0}": 1.0}, num_spin_orbitals=nqbit)
    # c_operators = {}
    # for k in range(1, nqbit):
    #     ck = FermionicOp({f"-_{k}": 1.0}, num_spin_orbitals=nqbit)
    #     c_aux = c0 @ ck
    #     c_operators[k] = qubit_converter.map(c_aux)


        # # --- cdagger Pair correlations ---
        # # --- Noise simulation ---
        # for k in range(1, nqbit):  # skip k=0
        #     job = estimator_QPU.run([(circuit, cdagger_operators[k])])
        #     result = job.result()
        #     cdaggercorrelations_QPU[q, k] = result[0].data.evs
        # # --- Exact simulation ---
        # for k in range(1, nqbit):  # skip k=0
        #     job = estimator_exact.run([(circuit, cdagger_operators[k])])
        #     result = job.result()
        #     cdaggercorrelations_exact[q, k] = result[0].data.evs
        # # --- c Pair correlations ---
        # # --- Noise simulation ---
        # for k in range(1, nqbit):  # skip k=0
        #     job = estimator_QPU.run([(circuit, c_operators[k])])
        #     result = job.result()
        #     ccorrelations_QPU[q, k] = result[0].data.evs
        # # --- Exact simulation ---
        # for k in range(1, nqbit):  # skip k=0
        #     job = estimator_exact.run([(circuit, c_operators[k])])
        #     result = job.result()
        #     ccorrelations_exact[q, k] = result[0].data.evs

    # if observable == 1:
    #     correlations_QPU = corr_results["majcorrelations_QPU"]
    #     correlations_exact = corr_results["majcorrelations_exact"]
    # elif observable == 2:
    #     correlations_QPU = corr_results["cdaggercorrelations_QPU"]
    #     correlations_exact = corr_results["cdaggercorrelations_exact"]
    # else: 
    #     correlations_QPU = corr_results["ccorrelations_QPU"]
    #     correlations_exact = corr_results["ccorrelations_exact"]