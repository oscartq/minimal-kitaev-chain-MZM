
import matplotlib as plt
import numpy as np

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
    theta_list     = np.asarray(results['theta_list'])           # (n_eps,)
    EnQ          = np.asarray(results['EnQ'])                # (n_eps, 2^N)
    Energy_BdG  = np.asarray(results['Energy_BdG'])        # (n_eps, 2^N)
    parity       = np.asarray(results['parity'])             # (n_eps, 2^N)
    nqbit        = results['nqbit']
    orb_comb    = results['orb_comb']
    eps_max     = float(theta_list.max())

    # Precompute Fock-state labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    # choose the index closest to 0+ to classify parity
    # (first index with eps >= 0; if none, use the one with smallest |eps|)
    if np.any(theta_list >= 0):
        i0p = int(np.argmax(theta_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(theta_list)))

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
            ax.plot(theta_list, y, color=color, linewidth=2.0, label=label)
            ax.annotate(f"{fock_labels[idx_vac]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor=color, color='w', fontsize=12)
        # Full -> (N-1)-particle excitations
        if occ == nqbit - 1:
            y = EnQ[:, idx_full] - EnQ[:, r]
            ax.plot(theta_list, y, color=color, linewidth=2.0, label=label)
            ax.annotate(f"{fock_labels[idx_full]}–{fock_labels[r]}",
                        xy=(eps_max, y[-1]),
                        backgroundcolor=color, color='w', fontsize=12)
        
    # --- Exact BdG eigenvalues as black "x" markers ---
    for r in range(0, 2**nqbit):
        ax.plot(theta_list, Energy_BdG,
                 marker='x', linestyle='None', color='black',
                label="BdG Hamiltonian eigenenergies")# if r == 0 else None)
        

    # --- styling consistent with other figures ---
    # dynamic limits with 5% padding based on both datasets
    y_min = np.min([EnQ.min(), Energy_BdG.min()])
    y_max = np.max([EnQ.max(), Energy_BdG.max()])
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(theta_list.min(), theta_list.max())
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varphi$', fontsize=18)

    # nicer ticks from the sweep rather than hard-coding
    xticks = [0, np.pi, 2*np.pi]
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title('Gaussian-state vs exact BdG spectrum')
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))              # keeps last occurrence
    ax.legend(uniq.values(), uniq.keys())

    plt.savefig("figures/plot_gaus_vs_diag_bdg.png")
    plt.show()

def plot_andreev_exact_diagonalization(results):
    """
    Plot full energy spectrum from exact diagonalization.
    
    Args:
        results: Dictionary containing simulation results
    """
    Energy_Diag = results['Energy_Diag']
    theta_list = results['theta_list']
    plt.figure(figsize=(10, 7))
    
    plt.plot(theta_list, Energy_Diag, linewidth=2.0)
    plt.ylim([np.min(Energy_Diag)*1.05, np.max(Energy_Diag)*1.05])
    plt.xlim([np.min(theta_list), np.max(theta_list)])

    plt.title("Exact diagonalization of many-body Hamiltonian")
    plt.ylabel(r'Energy', fontsize=18)
    plt.xlabel(r'$\varphi$', fontsize=18)
    plt.xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
    plt.savefig("figures/plot_exact_diagonalization.png")
    plt.show()

def plot_gaus_vs_diag_mb(results):
    """
    Plot comparison between Gaussian-state energies (EnQ)
    and exact many-body eigenvalues (Energy_Diag) across ε.
    Colors encode the parity at ε ≈ 0^+ (red: +1, blue: -1).
    """

    # --- unpack and cast once ---
    theta_list     = np.asarray(results['theta_list'])           # (n_eps,)
    EnQ          = np.asarray(results['EnQ'])                # (n_eps, 2^N)
    Energy_Diag  = np.asarray(results['Energy_Diag'])        # (n_eps, 2^N)
    parity       = np.asarray(results['parity'])             # (n_eps, 2^N)
    nqbit        = results['nqbit']

    # choose the index closest to 0+ to classify parity
    # (first index with eps >= 0; if none, use the one with smallest |eps|)
    if np.any(theta_list >= 0):
        i0p = int(np.argmax(theta_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(theta_list)))

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

        ax.plot(theta_list, EnQ[:, r], color=color, linewidth=2.0, label=label)

    # --- overlay exact many-body eigenvalues as black "x" markers ---
    for r in range(0, 2**nqbit):
        ax.plot(theta_list, Energy_Diag[:, r],
                marker='x', linestyle='None', color='black',
                label="Full diagonalization" if r == 0 else None)

    # --- styling consistent with other figures ---
    # dynamic limits with 5% padding based on both datasets
    y_min = np.min([EnQ.min(), Energy_Diag.min()])
    y_max = np.max([EnQ.max(), Energy_Diag.max()])
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(theta_list.min(), theta_list.max())
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varphi$', fontsize=18)

    # nicer ticks from the sweep rather than hard-coding
    xticks = [0, np.pi, 2*np.pi]
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title('Gaussian-state energies vs exact many-body spectrum')
    ax.legend()
    
    plt.savefig("figures/plot_gaus_vs_diag_mb.png")
    plt.show()