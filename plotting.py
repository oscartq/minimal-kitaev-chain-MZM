import numpy as np
import matplotlib.pyplot as plt
def plot_exact_diagonalization(results):
    """
    Plot full energy spectrum from exact diagonalization.
    
    Args:
        results: Dictionary containing simulation results
    """
    Energy_Diag = results['Energy_Diag']
    eps_list = results['eps_list']
    plt.figure(figsize=(10, 7))
    
    plt.plot(eps_list, Energy_Diag, linewidth=2.0)
    plt.ylim([np.min(Energy_Diag)*1.05, np.max(Energy_Diag)*1.05])
    plt.xlim([np.min(eps_list), np.max(eps_list)])

    plt.title("Exact diagonalization of many-body Hamiltonian")
    plt.ylabel(r'Energy', fontsize=18)
    plt.xlabel(r'$\varepsilon$', fontsize=18)
    plt.savefig("figures/plot_exact_diagonalization.png")
    plt.show()

def plot_bdg_comparison(results):

    # --------------------------
    # Helper: convert occupied-site list -> Fock state |n0 n1 ... nN-1>
    # --------------------------
    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    # --------------------------
    # Unpack results
    # --------------------------
    eps_list    = np.asarray(results['eps_list'])
    Energy_Diag = np.asarray(results['Energy_Diag'])
    Energy_BdG  = np.asarray(results['Energy_BdG'])     # shape: (n_eps, 2*N)
    orb_comb    = results['orb_comb']
    nqbit       = results['nqbit']
    eps_max     = float(eps_list.max())

    # Precompute Fock-state labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    # Index of vacuum and full-occupied configurations
    idx_vac  = 0
    idx_full = 2**nqbit - 1

    fig, ax = plt.subplots(figsize=(10, 7))

    # -----------------------------
    # BdG spectrum overlay
    # -----------------------------
    for band in range(Energy_BdG.shape[1]):
        ax.plot(eps_list, Energy_BdG[:, band], linewidth=2.0,
                linestyle='-',
                label="BdG Hamiltonian eigenenergies" if band == 0 else None,
                color="g")

    # -----------------------------
    # Many-body energy differences
    # -----------------------------
    for r in range(1, 2**nqbit - 1):
        occ = len(orb_comb[r])

        # Vacuum -> 1-particle excitations
        if occ == 1:
            y = Energy_Diag[:, idx_vac] - Energy_Diag[:, r]
            ax.plot(eps_list, y, color='red', linewidth=0.6, marker='x')
            ax.annotate(f"{fock_labels[idx_vac]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor="r", color='w', fontsize=12)

        # Full -> (N-1)-particle excitations
        if occ == nqbit - 1:
            y = Energy_Diag[:, idx_full] - Energy_Diag[:, r]
            ax.plot(eps_list, y, color='blue', linewidth=0.6, marker='x')
            ax.annotate(f"{fock_labels[idx_full]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor="b", color='w', fontsize=12)

    ax.legend()
    ax.set_title('Exact diagonalization of BdG Hamiltonian')
    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylim(Energy_BdG.min()*1.05, Energy_BdG.max()*1.05)
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)

    #xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks([eps_list.min(),0,eps_list.max()])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.savefig("figures/plot_bdg_comparison.png")
    plt.show()

def plot_gaus_vs_diag_mb(results):
    """
    Plot comparison between Gaussian-state energies (EnQ)
    and exact many-body eigenvalues (Energy_Diag) across ε.
    Colors encode the parity at ε ≈ 0^+ (red: +1, blue: -1).
    """

    # --- unpack and cast once ---
    eps_list     = np.asarray(results['eps_list'])           # (n_eps,)
    EnQ          = np.asarray(results['EnQ'])                # (n_eps, 2^N)
    Energy_Diag  = np.asarray(results['Energy_Diag'])        # (n_eps, 2^N)
    parity       = np.asarray(results['parity'])             # (n_eps, 2^N)
    nqbit        = results['nqbit']

    # choose the index closest to 0+ to classify parity
    # (first index with eps >= 0; if none, use the one with smallest |eps|)
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    # --- figure ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Legend flags so we only add two Gaussian legend entries (+1 and -1) once
    shown_plus = False
    shown_minus = False

    # --- plot EnQ curves colored by parity at ~0^+ ---
    for r in range(0, 2**nqbit):
        par = parity[i0p, r]
        color = 'red' if par > 0 else 'blue'

        label = None
        if par > 0 and not shown_plus:
            label = r'Gaussian state, $\langle\mathcal{P}(0^+)\rangle=+1$'
            shown_plus = True
        elif par <= 0 and not shown_minus:
            label = r'Gaussian state, $\langle\mathcal{P}(0^+)\rangle=-1$'
            shown_minus = True

        ax.plot(eps_list, EnQ[:, r], color=color, linewidth=2.0, label=label)

    # --- overlay exact many-body eigenvalues as black "x" markers ---
    for r in range(0, 2**nqbit):
        ax.plot(eps_list, Energy_Diag[:, r],
                marker='x', linestyle='None', color='black',
                label="Full diagonalization" if r == 0 else None)

    # --- styling consistent with other figures ---
    # dynamic limits with 5% padding based on both datasets
    y_min = np.min([EnQ.min(), Energy_Diag.min()])
    y_max = np.max([EnQ.max(), Energy_Diag.max()])
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)

    # nicer ticks from the sweep rather than hard-coding
    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title('Gaussian-state energies vs exact many-body spectrum')
    ax.legend()
    
    plt.savefig("figures/plot_gaus_vs_diag_mb.png")
    plt.show()

def plot_gaus_vs_diag_bdg(results):
    """
    Plot comparison between Gaussian-state energies (EnQ)
    and exact many-body eigenvalues (Energy_Diag) across ε.
    Colors encode the parity at ε ≈ 0^+ (red: +1, blue: -1).
    """
    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    # --- unpack and cast once ---
    eps_list     = np.asarray(results['eps_list'])           # (n_eps,)
    EnQ          = np.asarray(results['EnQ'])                # (n_eps, 2^N)
    Energy_BdG  = np.asarray(results['Energy_BdG'])        # (n_eps, 2^N)
    parity       = np.asarray(results['parity'])             # (n_eps, 2^N)
    nqbit        = results['nqbit']
    orb_comb    = results['orb_comb']
    eps_max     = float(eps_list.max())

    # Precompute Fock-state labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    # choose the index closest to 0+ to classify parity
    # (first index with eps >= 0; if none, use the one with smallest |eps|)
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    idx_vac  = 0
    idx_full = 2**nqbit - 1
    # --- figure ---
    fig, ax = plt.subplots(figsize=(10, 7))


    # Legend flags so we only add two Gaussian legend entries (+1 and -1) once
    shown_plus = False
    shown_minus = False
    # -----------------------------
    # Many-body energy differences
    # -----------------------------
    for r in range(1, 2**nqbit - 1):
        par = parity[i0p, r-1]
        color = 'red' if par > 0 else 'blue'

        label = None    
        if par > 0 and not shown_plus:
            label = r'Gaussian state, $\langle\mathcal{P}(0^+)\rangle=+1$'
            shown_plus = True
        elif par <= 0 and not shown_minus:
            label = r'Gaussian state, $\langle\mathcal{P}(0^+)\rangle=-1$'
            shown_minus = True

        occ = len(orb_comb[r])
        # Vacuum -> 1-particle excitations
        if occ == 1:
            y = EnQ[:, idx_vac] - EnQ[:, r]
            ax.plot(eps_list, y, color=color, linewidth=2.0, label=label)
            ax.annotate(f"{fock_labels[idx_vac]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor=color, color='w', fontsize=12)
        # Full -> (N-1)-particle excitations
        if occ == nqbit - 1:
            y = EnQ[:, idx_full] - EnQ[:, r]
            ax.plot(eps_list, y, color=color, linewidth=2.0, label=label)
            ax.annotate(f"{fock_labels[idx_full]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor=color, color='w', fontsize=12)
        
    # --- Exact BdG eigenvalues as black "x" markers ---
    for r in range(0, 2**nqbit):
        ax.plot(eps_list, Energy_BdG[:, r],
                marker='x', linestyle='None', color='black',
                label="BdG Hamiltonian eigenenergies" if r == 0 else None)
        

    # --- styling consistent with other figures ---
    # dynamic limits with 5% padding based on both datasets
    y_min = np.min([EnQ.min(), Energy_BdG.min()])
    y_max = np.max([EnQ.max(), Energy_BdG.max()])
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)

    # nicer ticks from the sweep rather than hard-coding
    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title('Gaussian-state vs exact BdG spectrum')
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))              # keeps last occurrence
    ax.legend(uniq.values(), uniq.keys())

    plt.savefig("figures/plot_gaus_vs_diag_bdg.png")
    plt.show()

def plot_parity(results):
    """
    Plot parity expectation values ⟨P⟩ for all many-body branches vs ε.
    Colors encode the parity at ε ≈ 0^+ (red: +1, blue: -1).
    """
    eps_list = np.asarray(results['eps_list'])     # (n_eps,)
    parity   = np.asarray(results['parity'])       # (n_eps, 2^N)
    nqbit    = results['nqbit']

    # choose the index closest to 0^+ to classify branch color
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    fig, ax = plt.subplots(figsize=(10, 7))

    shown_plus, shown_minus = False, False
    for r in range(0, 2**nqbit):
        par0 = parity[i0p, r]
        color = 'red' if par0 > 0 else 'blue'
        label = None
        if par0 > 0 and not shown_plus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=+1$'
            shown_plus = True
        elif par0 <= 0 and not shown_minus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=-1$'
            shown_minus = True

        ax.plot(eps_list, parity[:, r], color=color, linewidth=1.8, marker='x', label=label)

    # axes & styling
    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks(np.linspace(-1, 1, 5))
    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)

    ax.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax.set_ylabel(r'$\langle\mathcal{P}\rangle$', fontsize=18)
    ax.set_title('Fermion parity')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend()

    plt.savefig("figures/plot_parity.png")
    plt.show()

def plot_number_expectation(results):
    """
    Plot particle-number expectation ⟨N⟩ for all many-body branches vs ε.
    Colors encode parity at ε ≈ 0^+ (red: +1, blue: -1).
    Each curve is annotated at the right edge with its Fock-state label |n0 n1 ... nN-1⟩.
    """

    # helper: occupied-orbital list -> Fock label |n0 n1 ... nN-1>
    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    eps_list           = np.asarray(results['eps_list'])             # (n_eps,)
    number_expectation = np.asarray(results['number_expectation'])   # (n_eps, 2^N)
    parity             = np.asarray(results['parity'])               # (n_eps, 2^N)
    nqbit              = results['nqbit']
    orb_comb           = results.get('orb_comb', None)

    if orb_comb is None:
        raise ValueError("results must include 'orb_comb' so we can render Fock-state labels.")

    # index closest to 0^+ for parity-based coloring
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    # precompute labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    fig, ax = plt.subplots(figsize=(10, 7))

    shown_plus, shown_minus = False, False
    for r in range(0, 2**nqbit):
        par0 = parity[i0p, r]
        color = 'red' if par0 > 0 else 'blue'
        label = None
        if par0 > 0 and not shown_plus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=+1$'
            shown_plus = True
        elif par0 <= 0 and not shown_minus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=-1$'
            shown_minus = True

        ax.plot(eps_list, number_expectation[:, r], color=color, linewidth=1.8, marker='x', label=label)

        # annotate at the right edge
        ax.annotate(fock_labels[r],
                    xy=(eps_list[-1], number_expectation[-1, r]),
                    xytext=(2, 0), textcoords='offset points',
                    backgroundcolor=("r" if par0 > 0 else "b"),
                    color='w', fontsize=12)

    # axes & styling
    ax.set_xlim(eps_list.min(), eps_list.max())
    y_min, y_max = number_expectation.min(), number_expectation.max()
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax.set_ylabel(r'$\langle N \rangle$', fontsize=18)
    ax.set_title('Particle-number')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend()

    plt.savefig("figures/plot_number_expectation.png")
    plt.show()