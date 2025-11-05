# jj_phase_sweep_correlations.py
import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator

from functions import majorana_op, orbital_combinations, gauss_state_qiskit


def build_hamiltonian_jj(nqbit, eps, t=1.0, Delta=1.0, theta=0.0, tc=0.5):
    """
    Josephson junction of two minimal Kitaev dimers for nqbit=4.
      dimers: (0,1) with Δ*e^{-iθ/2}, (2,3) with Δ*e^{+iθ/2}
      inter-dimer: tc between sites 1↔2
      onsite: eps[i] on each site i

    Returns:
      ham_aux : FermionicOp
      circ_aux: string summary (passed to gauss_state_qiskit)
    """
    eps = np.asarray(eps, dtype=complex)
    if nqbit != 4:
        print(f"Warning: this Hamiltonian is written for nqbit=4, got {nqbit}.")
    if len(eps) != nqbit:
        raise ValueError(f"eps must have length {nqbit}, got {len(eps)}")

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
    return ham_aux, circ_aux


def compute_correlations_exact(nqbit, eps, t=1.0, Delta=1.0, theta=np.pi, tc=0.5):
    """
    Exact only. Returns:
      dict with correlations_exact [2**nqbit, 2*nqbit], orb_comb, orb_comb_str and params
    """
    eps = np.asarray(eps, dtype=complex)
    orb_comb = orbital_combinations(nqbit)
    orb_comb_str = [str(x) for x in orb_comb]

    # Hamiltonian (not directly used here, but we keep the same structure and circ string)
    ham_aux, circ_aux = build_hamiltonian_jj(nqbit, eps, t=t, Delta=Delta, theta=theta, tc=tc)

    # Build correlation operators i γ_1 γ_k with γ_0 being first Majorana on site 0
    nMaj = 2 * nqbit
    gamma0 = majorana_op(0, nqbit)
    qubit_converter = JordanWignerMapper()

    corr_ops_qubit = []
    for k in range(1, nMaj):
        gammak = majorana_op(k, nqbit)
        corr_fk = 1.0j * (gamma0 @ gammak)  # FermionicOp, Hermitian
        corr_ops_qubit.append(qubit_converter.map(corr_fk))

    estimator_exact = StatevectorEstimator()
    correlations_exact = np.full((2**nqbit, nMaj), np.nan, dtype=float)

    # Evaluate for all prepared Fock labels
    for q in range(2**nqbit):
        # Assuming gauss_state_qiskit returns (circuit, ...)
        circuit = gauss_state_qiskit(circ_aux, orb_comb[q], nqbit, eps)[0]
        # Batch the evaluations across k in one run
        pub_list = [(circuit, obs) for obs in corr_ops_qubit]
        job = estimator_exact.run(pub_list)
        res = job.result()
        # Fill columns k=1..2nqbit-1
        for k in range(1, nMaj):
            correlations_exact[q, k] = res[k-1].data.evs

    return {
        "correlations_exact": correlations_exact,
        "orb_comb": orb_comb,
        "orb_comb_str": orb_comb_str,
        "eps": eps,
        "nqbit": nqbit,
        "t": t,
        "Delta": Delta,
        "theta": theta,
        "tc": tc,
        "circ_aux": circ_aux,
    }


def plot_phase_grid(results_list, title_prefix="Majorana correlations vs k", savepath=None):
    """
    results_list: list of dicts from compute_correlations_exact for different thetas
    Plots a 10x2 grid. Each subplot corresponds to one theta.
    For each subplot, plots all Fock states:
      solid line for exact correlations ⟨i γ1 γk⟩, k from 1 to 2*nqbit-1
    """
    if len(results_list) != 20:
        print(f"Warning: expected 20 phase points for 10x2 grid, got {len(results_list)}")

    nqbit = results_list[0]["nqbit"]
    orb_comb = results_list[0]["orb_comb"]
    nMaj = 2 * nqbit
    k_values = np.arange(1, nMaj)
    n_states = 2**nqbit

    fig, axes = plt.subplots(10, 2, figsize=(12, 28), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        theta = res["theta"]
        corr = res["correlations_exact"]

        # choose colormap
        cm = plt.get_cmap('tab10') if n_states <= 10 else plt.get_cmap('gist_ncar')

        for q in range(n_states):
            color = cm(q / n_states) if n_states > 10 else cm(q)
            ax.plot(k_values, corr[q, 1:], linewidth=1.6, alpha=0.9, color=color)

        ax.set_title(rf"$\varphi={theta:.2f}$", fontsize=11)
        ax.grid(True, alpha=0.25)

    # Common axes labels and ticks
    for ax in axes:
        ax.set_xticks(k_values)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    fig.suptitle(title_prefix, fontsize=16)
    fig.text(0.5, 0.04, r"$k$", ha='center', fontsize=14)
    fig.text(0.04, 0.5, r"$\langle i\,\gamma_{1}\gamma_k \rangle$", va='center', rotation='vertical', fontsize=14)

    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.97])

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


def run_phase_sweep(
    nqbit=4,
    eps=(0.0, 0.0, 0.0, 0.0),
    t=1.0,
    Delta=1.0,
    tc=0.5,
    n_steps=20,
    endpoint=True,
    out_png="figures/jj_majorana_corr_phasegrid.png",
):
    """
    Sweeps θ in [0, 2π], computes exact correlations for each θ,
    and plots them in a 10x2 grid.
    """
    thetas = np.linspace(0.0, 2*np.pi, n_steps, endpoint=endpoint)
    results_list = []
    for th in thetas:
        res = compute_correlations_exact(
            nqbit=nqbit,
            eps=np.asarray(eps, dtype=complex),
            t=t,
            Delta=Delta,
            theta=th,
            tc=tc,
        )
        results_list.append(res)

    title = rf"Majorana correlations across phase, $\varepsilon={list(eps)}$, $t={t}$, $\Delta={Delta}$, $t_c={tc}$"
    plot_phase_grid(results_list, title_prefix=title, savepath=out_png)
    return results_list


if __name__ == "__main__":
    # Example run
    _ = run_phase_sweep(
        nqbit=4,
        eps=(0.0, 0.0, 0.0, 0.0),
        t=1.0,
        Delta=1.0,
        tc=0.5,
        n_steps=20,
        endpoint=True,  # includes 2π; set False if you prefer [0, 2π)
        out_png="figures/jj_majorana_corr_phasegrid.png",
    )