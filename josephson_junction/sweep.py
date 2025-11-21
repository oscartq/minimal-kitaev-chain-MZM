# jj_phase_sweep_correlations.py
import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator

from functions import majorana_op, orbital_combinations, gauss_state_qiskit


def build_hamiltonian_jj(nqbit, eps, t=1.0, Delta=1.0, theta=0.0, tc=0.5):
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
        ham_aux += FermionicOp({f"+_{j} -_{i}": np.conjugate(t)}, num_spin_orbitals=nqbit)
        circ_aux1 += f"{np.conjugate(t)}[{j}^ {i}]+"

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
    return ham_aux, circ_aux


def compute_correlations_exact_pair(nqbit, eps, t=1.0, Delta=1.0, theta=np.pi, tc=0.5):
    orb_comb = orbital_combinations(nqbit)
    orb_comb_str = [str(x) for x in orb_comb]

    # Build H and circ string (circuit uses circ_aux only)
    _, circ_aux = build_hamiltonian_jj(nqbit, eps, t=t, Delta=Delta, theta=theta, tc=tc)

    nMaj = 2 * nqbit
    gamma0 = majorana_op(0, nqbit)
    gamma4 = majorana_op(4, nqbit)

    qubit_converter = JordanWignerMapper()

    # Operators for γ0γk, k=1..4
    kvals_g0 = np.arange(1, 4, dtype=int)
    corr_ops_qubit_g0 = []
    for k in kvals_g0:
        corr_fk = 1.0j * (gamma0 @ majorana_op(k, nqbit))
        corr_ops_qubit_g0.append(qubit_converter.map(corr_fk))

    # Operators for γ4γk, k=5..nMaj-1
    kvals_g4 = np.arange(5, nMaj, dtype=int)
    corr_ops_qubit_g4 = []
    for k in kvals_g4:
        corr_fk = 1.0j * (gamma4 @ majorana_op(k, nqbit))
        corr_ops_qubit_g4.append(qubit_converter.map(corr_fk))

    estimator_exact = StatevectorEstimator()

    correlations_g0 = []
    correlations_g4 = []

    for q in range(2**nqbit):
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]

        job_g0 = estimator_exact.run([(circuit, corr_ops_qubit_g0)])
        result_g0 = job_g0.result()
        val_g0 = result_g0[0].data.evs
        correlations_g0.append(val_g0)

        job_g4 = estimator_exact.run([(circuit, corr_ops_qubit_g0)])
        result_g4 = job_g4.result()
        val_g4 = result_g4[0].data.evs
        correlations_g4.append(val_g4)

    return {
        "correlations_g0": correlations_g0,
        "correlations_g4": correlations_g4,
        "kvals_g0": kvals_g0,
        "kvals_g4": kvals_g4,
        "orb_comb": orb_comb,
        "orb_comb_str": orb_comb_str,
        "eps": eps,
        "nqbit": nqbit,
        "t": t,
        "Delta": Delta,
        "theta": theta,
        "tc": tc,
    }


def plot_three_phases_two_cols(results_list, title_prefix="Majorana correlations (φ ∈ {0, π, 2π})", savepath=None):

    nqbit = results_list[0]["nqbit"]
    n_states = 2**nqbit

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False, sharey=True)
    phases_labels = []
    for idx, res in enumerate(results_list):
        phases_labels.append(res["theta"])

        # choose colormap per subplot for readability
        cm = plt.get_cmap('tab10') if n_states <= 10 else plt.get_cmap('gist_ncar')

        # Left column: γ0γk
        axL = axes[idx, 0]
        kvals_g0 = res["kvals_g0"]
        corr_g0 = res["correlations_g0"]
        for q in range(n_states):
            color = cm(q / n_states) if n_states > 10 else cm(q)
            axL.plot(kvals_g0, corr_g0[q, :], linewidth=1.6, alpha=0.9, color=color)
        axL.set_title(rf"$\varphi={res['theta']:.2f}$,  $i\gamma_0\gamma_k$", fontsize=11)
        axL.set_xticks(kvals_g0)
        axL.grid(True, alpha=0.25)

        # Right column: γ4γk
        axR = axes[idx, 1]
        kvals_g4 = res["kvals_g4"]
        corr_g4 = res["correlations_g4"]
        for q in range(n_states):
            color = cm(q / n_states) if n_states > 10 else cm(q)
            axR.plot(kvals_g4, corr_g4[q, :], linewidth=1.6, alpha=0.9, color=color)
        axR.set_title(rf"$i\gamma_4\gamma_k$", fontsize=11)
        axR.set_xticks(kvals_g4)
        axR.grid(True, alpha=0.25)

        # Tick sizes
        axL.tick_params(axis='x', labelsize=9)
        axL.tick_params(axis='y', labelsize=9)
        axR.tick_params(axis='x', labelsize=9)
        axR.tick_params(axis='y', labelsize=9)

    # Common labels
    fig.suptitle(title_prefix, fontsize=16)
    fig.text(0.5, 0.04, r"$k$", ha='center', fontsize=14)
    fig.text(0.04, 0.5, r"$\langle i\,\gamma_{j}\gamma_k \rangle$", va='center', rotation='vertical', fontsize=14)

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.97])

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()

def run_three_phases(
    nqbit=4,
    eps=[0.0, 0.0, 0.0, 0.0],
    t=1.0,
    Delta=1.0,
    tc=0.5,
    out_png="figures/jj_majorana_corr_threephases_3x2.png",
):
    phases = [0.0, np.pi, 2*np.pi]
    results_list = []
    for th in phases:
        res = compute_correlations_exact_pair(
            nqbit=nqbit,
            eps=np.asarray(eps, dtype=complex),
            t=t,
            Delta=Delta,
            theta=th,
            tc=tc,
        )
        results_list.append(res)

    title = rf"Majorana correlations, $\varepsilon={list(eps)}$, $t={t}$, $\Delta={Delta}$, $t_c={tc}$"
    plot_three_phases_two_cols(results_list, title_prefix=title, savepath=out_png)
    return results_list

if __name__ == "__main__":
    _ = run_three_phases(
        nqbit=4,
        eps=(0.0, 0.0, 0.0, 0.0),
        t=1.0,
        Delta=1.0,
        tc=1.0,
        out_png="figures/jj_majorana_corr_threephases_3x2.png",
    )
