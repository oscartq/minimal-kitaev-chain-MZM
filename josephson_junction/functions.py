import numpy as np
import openfermion
from openfermion.ops import FermionOperator
from qiskit_nature.second_q.operators import FermionicOp
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
        >>> orbital_combinations(2)
        [[], [0], [1], [0, 1]]
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

def compute_bdg_spectrum(nqbit, nsteps,eps, t, Delta,tc):
    Energy_BdG = []
    Fermi_En_Pl = []
    Fermi_En_Mi = []

    dtheta = 2*np.pi / nsteps  # step size

    for k in range(0, nsteps + 1):
        theta = k * dtheta
        t_L = t
        t_R = t
        t_L_star = np.conjugate(t_L)
        t_R_star = np.conjugate(t_R)
        Delta_L = Delta*np.exp(-1.0j*theta/2)
        Delta_R = Delta*np.exp(1.0j*theta/2)
        Delta_L_star = np.conjugate(Delta_L)
        Delta_R_star = np.conjugate(Delta_R)

        epsilon1, epsilon2, epsilon3, epsilon4 = [0, 0, 0, 0]
        M = np.array([
        [epsilon1,      t_L,           0,             0,             0,               Delta_L,         0,               0          ],
        [t_L_star,      epsilon2,      tc,           0,             -Delta_L,        0,               0,               0          ],
        [0,             tc,           epsilon3,      t_R,           0,               0,               0,               Delta_R    ],
        [0,             0,             t_R_star,      epsilon4,      0,               0,               -Delta_R,        0          ],
        [0,             -Delta_L_star, 0,             0,             -epsilon1,       -t_L_star,       0,               0          ],
        [Delta_L_star,  0,             0,             0,             -t_L,            -epsilon2,       -tc,            0          ],
        [0,             0,             0,             -Delta_R_star, 0,               -tc,            -epsilon3,       -t_R_star  ],
        [0,             0,             Delta_R_star,  0,             0,               0,               -t_R,            -epsilon4  ]
        ])

        # eigenvalues
        Energy_BdG.append(np.linalg.eigh(M)[0])

        # Fermi energy offsets (same convention as before)
        Fermi_En_Pl.append(epsilon1 * (nqbit / 2))
        Fermi_En_Mi.append(-epsilon1 * (nqbit / 2))

    return Energy_BdG, Fermi_En_Pl, Fermi_En_Mi