"""
Utility functions for Gaussian state preparation and BCS spectrum computation.
Updated for modern Qiskit versions (1.0+).
"""

import numpy as np
import openfermion
from openfermion.ops import FermionOperator
from openfermion import gaussian_state_preparation_circuit
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumRegister, QuantumCircuit
from itertools import combinations


def orbital_combinations(nqbit):
    """
    Generate all possible orbital occupation combinations sorted by total occupation sum.
    Example:
        >>> orbital_combinations(3)
        [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    """
    count_ar = []
    aux_ar = []
    
    # Generate list of orbital indices
    for i in range(0, nqbit):
        count_ar.append(i)
    
    # Generate all possible combinations of occupied orbitals
    for i in range(0, nqbit + 1):
        a = combinations(count_ar, i)
        b = list(a)
        for j in range(0, len(b)):
            aux_ar.append(list(b[j]))
    
    ordering = []
    
    # Sort combinations by sum of orbital indices
    for i in range(0, sum(count_ar) + 1):
        for j in range(0, len(aux_ar)):
            if (sum(aux_ar[j]) == i):
                ordering.append(aux_ar[j])
    
    return ordering


def YXXY(fi):
    """
    Define the custom RYXXY gate.
    
    Args:
        fi: Rotation angle parameter
        
    Returns:
        UnitaryGate: Custom two-qubit unitary gate
    """
    arr = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(fi), np.sin(fi), 0.0],
        [0.0, -np.sin(fi), np.cos(fi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    u_gate = UnitaryGate(data=arr, label='RYXXY(' + str(fi) + ")")
    return u_gate


def gauss_state_qiskit(circ_aux, OccupOrb, nqbit, eps):
    """
    Prepare a Gaussian state in Qiskit using OpenFermion's circuit generation.
    
    Args:
        circ_aux: String representation of the fermionic Hamiltonian
        OccupOrb: List of initially occupied orbitals
        nqbit: Number of qubits
        eps: Chemical potential parameter
        
    Returns:
        tuple: (QuantumCircuit, OccupOrb, nqbit, eps)
    """
    # Parse the Hamiltonian string and create FermionOperator
    kitaev_model = FermionOperator(circ_aux[0:len(circ_aux)-1])
    
    # Convert to quadratic Hamiltonian
    quad_ham = openfermion.get_quadratic_hamiltonian(
        kitaev_model, 
        ignore_incompatible_terms=True
    )
    
    # Generate Gaussian state preparation circuit using OpenFermion
    circuit_description, start_orbitals = gaussian_state_preparation_circuit(
        quad_ham, 
        occupied_orbitals=OccupOrb, 
        spin_sector=None
    )
    
    nr_var_par = 0
    var_ar = []
    
    # Create Qiskit circuit
    qr = QuantumRegister(nqbit, 'q')
    qc = QuantumCircuit(qr)
    
    # Initialize occupied orbitals
    for i in start_orbitals:
        qc.x(qr[i])
    
    # Translate circuit description to Qiskit gates
    for i in range(0, len(circuit_description)):
        for j in range(0, len(circuit_description[i])):
            if type(circuit_description[i][j]) == tuple:
                # Apply custom YXXY gate
                qc.append(
                    YXXY(circuit_description[i][j][2]), 
                    [qr[int(circuit_description[i][j][0])], 
                     qr[int(circuit_description[i][j][1])]]
                )
                # Apply RZ rotation
                qc.rz(
                    circuit_description[i][j][3], 
                    qr[int(circuit_description[i][j][1])]
                )
                nr_var_par += 2
                var_ar.append(circuit_description[i][j][2])
                var_ar.append(circuit_description[i][j][3])
            else:
                # Apply X gate to last qubit
                qc.x(qr[nqbit-1])
    
    return qc, OccupOrb, nqbit, eps


def compute_bdg_spectrum(nqbit, nsteps, eps_max, t, Delta, const=1.0):
    Energy_BdG = []
    Fermi_En_Pl = []
    Fermi_En_Mi = []

    deps = eps_max / nsteps  # step size
    # symmetric sweep: -eps_max ... 0 ... +eps_max
    for k in range(-nsteps, nsteps + 1):
        eps = k * deps

        # single-particle H and pairing Δ
        ham_aux_2 = np.zeros((nqbit, nqbit), dtype=complex)  # H (hopping + chemical potential)
        ham_aux_3 = np.zeros((nqbit, nqbit), dtype=complex)  # Δ (pairing)

        for i in range(0, nqbit - 1):
            # hopping
            ham_aux_2[i, i + 1] = -const * t
            ham_aux_2[i + 1, i] = -const * t

            # on-site (chemical potential)
            ham_aux_2[i, i] = -eps

            # nearest-neighbor pairing
            ham_aux_3[i, i + 1] = const * Delta
            ham_aux_3[i + 1, i] = -const * np.conj(Delta)

        # last site's chemical potential
        ham_aux_2[nqbit - 1, nqbit - 1] = -eps

        # BdG Hamiltonian: H_BdG = σ_z ⊗ H + σ_y' ⊗ Δ  with σ_y'=[[0,1],[-1,0]]
        ham_aux_4 = (np.kron(np.diag([1, -1]), ham_aux_2) +
                     np.kron(np.array([[0, 1], [-1, 0]]), ham_aux_3))

        # eigenvalues
        Energy_BdG.append(np.linalg.eigh(ham_aux_4)[0])

        # Fermi energy offsets (same convention as before)
        Fermi_En_Pl.append(eps * (nqbit / 2))
        Fermi_En_Mi.append(-eps * (nqbit / 2))

    return Energy_BdG, Fermi_En_Pl, Fermi_En_Mi
