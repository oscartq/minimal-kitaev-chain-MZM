import numpy as np
import matplotlib.pyplot as plt

def plot_correlation(results):
    """
    Plot end-to-end Majorana correlation vs ε for each many-body branch.
    Colors encode parity at ε ≈ 0^+ (red: +1, blue: -1).
    Labels use Fock-state notation |n0 n1 ... nN-1>.
    """
    # helper: occupied-orbital list -> Fock label |n0 n1 ... nN-1>
    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    # unpack and cast once
    eps_list    = np.asarray(results['eps_list'])       # (n_eps,)
    corr        = np.asarray(results['correlation'])    # (n_eps, 2^N)  end-to-end iγ1γ_{2N}
    parity      = np.asarray(results['parity'])         # (n_eps, 2^N)
    nqbit       = results['nqbit']
    orb_comb    = results.get('orb_comb', None)

    if orb_comb is None:
        raise ValueError("results must include 'orb_comb' so we can render Fock-state labels.")

    # parity classification at ε ≈ 0^+
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    # precompute Fock labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    fig, ax = plt.subplots(figsize=(10, 7))

    shown_plus, shown_minus = False, False

    # plot each branch colored by parity at ~0^+
    for r in range(0, 2**nqbit):
        par = parity[i0p, r]
        color = 'red' if par > 0 else 'blue'

        label = None
        if par > 0 and not shown_plus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=+1$'
            shown_plus = True
        elif par <= 0 and not shown_minus:
            label = r'$\langle\mathcal{P}(0^+)\rangle=-1$'
            shown_minus = True

        ax.plot(eps_list, corr[:, r], color=color, linewidth=1.5, marker='x', label=label)

        # annotate near ε≈0 and at right edge
        # pick the index closest to 0 and the last point
        i_zero = int(np.argmin(np.abs(eps_list)))
        ax.annotate(fock_labels[r],
                    xy=(eps_list[i_zero], corr[i_zero, r]),
                    xytext=(2, 0), textcoords='offset points',
                    backgroundcolor=("r" if par > 0 else "b"),
                    color='w', fontsize=11)
        ax.annotate(fock_labels[r],
                    xy=(eps_list[-1], corr[-1, r]),
                    xytext=(2, 0), textcoords='offset points',
                    backgroundcolor=("r" if par > 0 else "b"),
                    color='w', fontsize=11)

    # styling
    ax.set_xlim(eps_list.min(), eps_list.max())
    y_min, y_max = corr.min(), corr.max()
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    # dynamic ticks
    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(-1, 1, 5))

    # labels and title
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax.set_ylabel(rf'$\langle i\,\gamma_{{1}}\gamma_{{{2*nqbit}}} \rangle$', fontsize=18)
    ax.set_title('End-to-end Majorana correlation by many-body branch')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend()
    plt.show()


def plot_QPU_many_body(results):
    """
    Plot many-body energy spectrum EnQ with Fock-state labels.
    Colors encode parity at ε ≈ 0^+ (red: +1, blue: -1).
    """

    # helper: occupied-orbital list -> Fock label |n0 n1 ... nN-1>
    def fock_label(occ_list, nqbit):
        bits = ['1' if i in occ_list else '0' for i in range(nqbit)]
        return r"$|" + ''.join(bits) + r" \rangle$"

    # unpack and cast once
    eps_list     = np.asarray(results['eps_list'])       # (n_eps,)
    EnQ          = np.asarray(results['EnQ'])            # (n_eps, 2^N)
    parity       = np.asarray(results['parity'])         # (n_eps, 2^N)
    nqbit        = results['nqbit']
    orb_comb     = results.get('orb_comb', None)         # list of lists (occupied sites)
    orb_comb_str = results.get('orb_comb_str', None)     # fallback labels if needed

    if orb_comb is None:
        raise ValueError("results must include 'orb_comb' so we can render Fock-state labels.")

    # index closest to 0^+ to classify parity
    if np.any(eps_list >= 0):
        i0p = int(np.argmax(eps_list >= 0))
    else:
        i0p = int(np.argmin(np.abs(eps_list)))

    # precompute Fock labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    fig, ax = plt.subplots(figsize=(10, 7))

    # legend flags so we only add the two parity entries once
    shown_plus, shown_minus = False, False

    # plot each many-body branch, colored by parity at ~0^+
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

        ax.plot(eps_list, EnQ[:, r], color=color, marker='x', linewidth=0.8, label=label)

        # annotate at the right edge (last ε in the sweep)
        ax.annotate(fock_labels[r],
                    xy=(eps_list[-1], EnQ[-1, r]),
                    xytext=(2, 0), textcoords='offset points',
                    backgroundcolor=("r" if par > 0 else "b"),
                    color='w', fontsize=12)

    # style to match other figures
    y_min = EnQ.min()
    y_max = EnQ.max()
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)

    xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title('Gaussian states many-body energy spectrum ')
    ax.legend()
    plt.show()

def plot_QPU_bdg(results):

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
    Energy_Diag = np.asarray(results['EnQ'])
    Energy_BdG  = np.asarray(results['Energy_BdG'])     # shape: (n_eps, 2*N)
    orb_comb    = results['orb_comb']
    nqbit       = results['nqbit']
    eps_max     = float(eps_list.max())

    # Precompute Fock-state labels
    fock_labels = [fock_label(occ, nqbit) for occ in orb_comb]

    # Index of vacuum and full-occupied configurations
    idx_vac  = 0
    idx_full = 2**nqbit - 1

    fig, ax = plt.subplots(figsize=(8, 6))

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
    ax.set_title('Exact vs noisy BdG Hamiltonian')
    ax.set_xlim(eps_list.min(), eps_list.max())
    ax.set_ylim(Energy_BdG.min()*1.05, Energy_BdG.max()*1.05)
    ax.set_ylabel(r'Energy', fontsize=18)
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)

    #xticks = np.linspace(eps_list.min(), eps_list.max(), 7)
    ax.set_xticks([eps_list.min(),0,eps_list.max()])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.show()