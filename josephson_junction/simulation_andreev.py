import numpy as np
import time

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Session

from functions import gauss_state_qiskit, orbital_combinations, majorana_op, compute_bdg_spectrum

def run_simulation(nqbit=4, nsteps=20, eps=[0,0,0,0], t=1.0, Delta=1.0, tc=1.0):
    
    t1 = time.time()

    # Create Estimator instance (for exact statevector simulation)
    estimator = StatevectorEstimator()
    # backend = FakeBrisbane()
    # estimator = BackendEstimatorV2(backend=backend)

    print(rf"Starting simulation with nqbit={nqbit}, theta range: 0 to $2\pi$, nsteps={nsteps}")
    print(f"Total number of theta points: {nsteps}")
    
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
    theta_list = []

    # symmetric sweep: -eps_max ... 0 ... +eps_max in 2*nsteps+1 points
    theta_max = 2*np.pi
    dtheta = theta_max / nsteps
    total_pts = nsteps + 1

    for idx, k in enumerate(range(0, nsteps+1), start=1):
        print(f"Executing step {idx}/{total_pts} (k={k})")
        theta = k * dtheta
        theta_list.append(theta)

        Delta_aux = [Delta*np.exp(-1.0j*theta/2),Delta*np.exp(1.0j*theta/2)]
        # This is the actual Hamiltonian in use.
        # ---------- Josephson junction of two minimal Kitaev chains ---------- 
        ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
        circ_aux1, circ_aux2 = "", ""

        # -------- Intra-dimer terms: (0,1) with t[0], Δ[0]; (2,3] with t[1], Δ[1] --------
        dimers = [(0, 1), (2, 3)]
        for h, (i, j) in enumerate(dimers):
            # Hopping i<->j
            ham_aux += FermionicOp({f"+_{i} -_{j}": t}, num_spin_orbitals=nqbit)
            circ_aux1 += f"{t}[{i}^ {j}]+"
            ham_aux += FermionicOp({f"+_{j} -_{i}": t}, num_spin_orbitals=nqbit)
            circ_aux1 += f"{t}[{j}^ {i}]+"

            # Pairing +i +j and -j -i
            ham_aux += FermionicOp({f"+_{i} +_{j}": Delta_aux[h]}, num_spin_orbitals=nqbit)
            circ_aux1 += f"{Delta_aux[h]}[{i}^ {j}^]+"
            ham_aux += FermionicOp({f"-_{j} -_{i}": np.conjugate(Delta_aux[h])}, num_spin_orbitals=nqbit)
            circ_aux1 += f"{np.conjugate(Delta_aux[h])}[{j} {i}]+"

        # -------- Inter-dimer coupling between sites 1 and 2 --------
        # Follow your original: add tc in both directions (use conjugate if you want strict Hermiticity for complex tc)
        ham_aux += FermionicOp({f"+_{1} -_{2}": tc}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{tc}[1^ 2]+"
        ham_aux += FermionicOp({f"+_{2} -_{1}": np.conjugate(tc)}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{np.conjugate(tc)}[2^ 1]+"

        # -------- Onsite energies --------
        for i in range(nqbit):
            ham_aux += FermionicOp({f"+_{i} -_{i}": eps[i]}, num_spin_orbitals=nqbit)
            circ_aux2 += f"{eps[i]}[{i}^ {i}]+"

        circ_aux = circ_aux1 + circ_aux2  # if you want to keep the string summary

        # -------- Jordan–Wigner mapping to Pauli operators --------
        jw = JordanWignerMapper()
        pauli_H = jw.map(ham_aux)
        hamilt_JW = qubit_converter.map(ham_aux)
        
        # Majorana operator i\gamma_1\gamma_N
        corr = \
            FermionicOp({f"+_0 +_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            + FermionicOp({f"-_0 +_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            - FermionicOp({f"+_0 -_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit) \
            - FermionicOp({f"-_0 -_{nqbit-1}": 1.0}, num_spin_orbitals=nqbit)
        
        corr_JW = qubit_converter.map(corr)
        
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
            val = result[0].data.evs #- eps[0] * nqbit / 2 #Minus the onsite energy offset
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
        Energy_Diag.append(list(np.linalg.eigh(hamilt_JW.to_matrix())[0] - [eps[0]*(nqbit/2)]*2**nqbit)) #Diagonalization of the normal Hamiltonian
    
    Energy_BdG, Fermi_En_Pl, Fermi_En_Mi = compute_bdg_spectrum(nqbit, nsteps, eps, t, Delta, tc)
    
    t2 = time.time()
    print(f"\nSimulation completed in {t2-t1:.2f} seconds")
    
    # Return results as dictionary
    results = {
        'Energy_Diag': Energy_Diag,
        'EnQ': EnQ,
        'correlation': correlation,
        'number_expectation': number_expectation,
        'parity': parity,
        'theta_list': theta_list,
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

def compute_correlations(nqbit,eps,t=1.0,Delta=1.0,theta=np.pi,tc=0.5):             
    """
    Josephson junction of two minimal Kitaev dimers:
      dimers: (0,1) with Δ*e^{-iφ/2}, (2,3) with Δ*e^{+iφ/2}
      inter-dimer: tc between sites 1↔2
      onsite: eps[i] on each site i

    Returns correlation matrices for i γ_1 γ_k (k=1..2*nqbit-1) for all prepared Fock states.
    """

    # ---- basic checks ----
    eps = np.asarray(eps, dtype=complex)
    if nqbit != 4:
        print(f"Warning: this Hamiltonian is written for nqbit=4 (two dimers), got {nqbit}.")
    if len(eps) != nqbit:
        raise ValueError(f"eps must have length {nqbit}, got {len(eps)}")

    print(f"Computing correlations (JJ) for nqbit={nqbit}, eps={eps}, t={t}, Δ={Delta}, φ={theta}, tc={tc}")

    backend = FakeBrisbane()
    estimator_QPU = BackendEstimatorV2(backend=backend)
    estimator_exact = StatevectorEstimator()

    # Prepare orbital combinations (you already have these helpers)
    orb_comb = orbital_combinations(nqbit)
    orb_comb_str = [str(x) for x in orb_comb]

    qubit_converter = JordanWignerMapper()

    # ---- Build Hamiltonian (FermionicOp) and a string summary ----
    ham_aux = FermionicOp({"": 0.0}, num_spin_orbitals=nqbit)
    circ_aux1, circ_aux2 = "", ""

    # Superconducting phases on the two dimers
    Delta_aux = [Delta * np.exp(-1.0j * theta / 2.0),
                 Delta * np.exp(+1.0j * theta / 2.0)]

    # Intra-dimer terms: (0,1) uses Delta_aux[0], (2,3) uses Delta_aux[1]
    dimers = [(0, 1), (2, 3)]
    for h, (i, j) in enumerate(dimers):
        # hopping i<->j
        ham_aux += FermionicOp({f"+_{i} -_{j}": t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{t}[{i}^ {j}]+"
        ham_aux += FermionicOp({f"+_{j} -_{i}": t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{t}[{j}^ {i}]+"

        # pairing +i +j and -j -i with the complex phase
        ham_aux += FermionicOp({f"+_{i} +_{j}": Delta_aux[h]}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{Delta_aux[h]}[{i}^ {j}^]+"
        ham_aux += FermionicOp({f"-_{j} -_{i}": np.conjugate(Delta_aux[h])}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{np.conjugate(Delta_aux[h])}[{j} {i}]+"

    # Inter-dimer coupling between sites 1 and 2 (allow complex tc)
    ham_aux += FermionicOp({f"+_{1} -_{2}": tc}, num_spin_orbitals=nqbit)
    circ_aux1 += f"{tc}[1^ 2]+"
    ham_aux += FermionicOp({f"+_{2} -_{1}": np.conjugate(tc)}, num_spin_orbitals=nqbit)
    circ_aux1 += f"{np.conjugate(tc)}[2^ 1]+"

    # Onsite energies
    for i in range(nqbit):
        ham_aux += FermionicOp({f"+_{i} -_{i}": eps[i]}, num_spin_orbitals=nqbit)
        circ_aux2 += f"{eps[i]}[{i}^ {i}]+"

    circ_aux = circ_aux1 + circ_aux2

    # ---- correlation operators: i γ_1 γ_k with 0-based indices (γ_0 ≡ first Majorana on site 0) ----
    gamma0 = majorana_op(0, nqbit)

    corr_operators = {}
    nMaj = 2 * nqbit
    for k in range(1, nMaj):
        gammak = majorana_op(k, nqbit)
        corr = 1.0j * (gamma0 @ gammak)  # Hermitian for k>0
        corr_operators[k] = qubit_converter.map(corr)

    # ---- evaluate for all prepared Fock labels from orb_comb ----
    correlations_QPU = np.full((2**nqbit, nMaj), np.nan, dtype=float)
    correlations_exact = np.full((2**nqbit, nMaj), np.nan, dtype=float)

    for q in range(2**nqbit):
        print(f"  Computing state {q+1}/{2**nqbit}: {orb_comb_str[q]}")
        # Assuming your gauss_state_qiskit accepts (circ_aux, occ_list, nqbit, eps)
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]

        # Backends
        for k in range(1, nMaj):
            job = estimator_QPU.run([(circuit, corr_operators[k])])
            result = job.result()
            correlations_QPU[q, k] = result[0].data.evs

        for k in range(1, nMaj):
            job = estimator_exact.run([(circuit, corr_operators[k])])
            result = job.result()
            correlations_exact[q, k] = result[0].data.evs

    print("Correlation computation complete!")

    return {
        "correlations_QPU": correlations_QPU,
        "correlations_exact": correlations_exact,
        "orb_comb": orb_comb,
        "orb_comb_str": orb_comb_str,
        "eps": eps,
        "nqbit": nqbit,
        "t": t,
        "Delta": Delta,
        "theta": theta,
        "tc": tc,
        "circ_aux": circ_aux,  # optional: may help debug/record
    }

def plot_correlation_vs_k(corr_results, state_indices=None,):
    import numpy as np
    import matplotlib.pyplot as plt

    correlations_QPU = corr_results["correlations_QPU"]
    correlations_exact = corr_results["correlations_exact"]
    orb_comb = corr_results["orb_comb"]
    nqbit = corr_results["nqbit"]
    eps = corr_results["eps"]
    theta = corr_results.get("theta", 0.0)

    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]
    if state_indices is None:
        state_indices = np.arange(2**nqbit)

    nMaj = correlations_exact.shape[1]  # = 2*nqbit
    k_max = nMaj - 1  # we plot k = 1..k_max
    k_values = np.arange(1, k_max + 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    NUM_COLORS = len(state_indices)
    cm = plt.get_cmap('tab10') if NUM_COLORS <= 10 else plt.get_cmap('gist_ncar')

    for idx, q in enumerate(state_indices):
        color = cm(idx / NUM_COLORS) if NUM_COLORS > 10 else cm(idx)
        # slice columns [1..k_max]
        ax.plot(k_values, correlations_QPU[q, 1:k_max + 1],
                marker="*", linestyle="None", color=color)
        ax.plot(k_values, correlations_exact[q, 1:k_max + 1],
                label=f"{fock_labels[q]}", linewidth=2, color=color)

    # Legend proxies
    ax.plot([], [], color="black", linestyle="-", label="Exact")
    ax.plot([], [], color="black", marker="*", linestyle="None", label="QPU")

    ax.set_ylabel(r'$\langle i\,\gamma_{1}\gamma_k \rangle$', fontsize=18)
    ax.set_xlabel(r'$k$', fontsize=18)

    eps_str = "[" + ", ".join(f"{complex(e):.2f}" if np.iscomplexobj(e) else f"{float(e):.2f}"
                              for e in np.asarray(eps)) + "]"
    ax.set_title(rf'Majorana correlations, $\varphi={theta:.2f}$ rad, $\varepsilon={eps_str}$',
                 fontsize=16)

    ax.set_xticks(k_values)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.3)

    # filename includes phase and eps compactly
    eps_fn = "_".join(f"{float(np.real(x)):.2f}" if np.isreal(x) else f"{np.real(x):.2f}+{np.imag(x):.2f}i"
                      for x in np.asarray(eps))
    fig.savefig(f"figures/plot_correlation_phi={theta:.2f}_eps={eps_fn}.png", dpi=200, bbox_inches="tight")
    plt.show()

# energy = compute_bdg_spectrum(4, 20, [0,0,0,0], 1, 1, 1)
# print(len(energy[0,0]))
# if __name__ == "__main__":
#     # Run with default parameters
#     results = run_simulation()
    
#     # Optionally save results
#     with open('simulation_andreev_results.pkl', 'wb') as f:
#         pickle.dump(results, f)
#     print("\nResults saved to 'simulation_andreev_results.pkl'")