import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from functions import gauss_state_qiskit, orbital_combinations

# --- helper: build a single Majorana operator γ_m (0-based m in [0, 2*nqbit-1]) ---
def majorana_op(m, nqbit):
    j = m // 2  # site index (0..nqbit-1)
    if m % 2 == 0:
        # m = 0,2,4,... -> γ_{2j-1} = a_j + a_j^\dagger
        return (FermionicOp({f"-_{j}": 1.0}, num_spin_orbitals=nqbit)
                + FermionicOp({f"+_{j}": 1.0}, num_spin_orbitals=nqbit))
    else:
        # m = 1,3,5,... -> γ_{2j} = -i(a_j - a_j^\dagger) = (-i) a_j + (i) a_j^\dagger
        return (-1.0j) * FermionicOp({f"-_{j}": 1.0}, num_spin_orbitals=nqbit) \
               + (1.0j) * FermionicOp({f"+_{j}": 1.0}, num_spin_orbitals=nqbit)

def compute_correlations(nqbit, eps, t=-1.0, Delta=1.0, const=1.0):

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
        circ_aux1 += f"{-t*const}[{i}^ {i+1}]+"
        ham_aux += FermionicOp({f"+_{i+1} -_{i}": -t}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{-t*const}[{i+1}^ {i}]+"
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

    corr_operators = {}
    nMaj = 2 * nqbit
    for k in range(1, nMaj):
        gammak = majorana_op(k, nqbit)
        corr = 1.0j * (gamma0 @ gammak)  # this is Hermitian for k>0
        corr_operators[k] = qubit_converter.map(corr)

    # --- allocate full [state, k] with k over 2*nqbit ---
    correlations_QPU = np.full((2**nqbit, nMaj), np.nan, dtype=float)
    correlations_exact = np.full((2**nqbit, nMaj), np.nan, dtype=float)

    for q in range(2**nqbit):
        print(f"  Computing state {q+1}/{2**nqbit}: {orb_comb_str[q]}")
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]

        for k in range(1, nMaj):  # skip k=0
            job = estimator_QPU.run([(circuit, corr_operators[k])])
            result = job.result()
            correlations_QPU[q, k] = result[0].data.evs

        for k in range(1, nMaj):  # skip k=0
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
