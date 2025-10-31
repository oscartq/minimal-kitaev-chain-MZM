# compute_matrix.py
from sympy import symbols, Matrix, I, conjugate, pprint, simplify

# Symbols
eps1, eps2, t, Delta = symbols('eps1 eps2 t Delta', complex=True)

# A matrix (use I for sqrt(-1))
A = Matrix([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [I, 0, -I, 0],
    [0, I, 0, -I],
])

# B matrix
B = Matrix([
    [0,        Delta,   eps1,  t    ],
    [-Delta,   0,       t,     eps2 ],
    [-eps1,   -t,       0,    -Delta],
    [-t,      -eps2,    Delta, 0    ],
])

# Compute: -i * conjugate(A) * B * conjugate_transpose(A)
M = -I * A.conjugate() * B * A.conjugate().transpose()  # or A.H for conjugate transpose

# Optionally simplify
M_simplified = simplify(M)

print("Resulting matrix M = -i * conj(A) * B * conj(A)^T:")
pprint(M_simplified, use_unicode=True)

# If you prefer a plain array-like print:
# print(M_simplified)

# Example: plug in numbers (uncomment to test)
# subs_M = M_simplified.subs({eps1: 1.2, eps2: -0.7, t: 0.5, Delta: 0.3})
# pprint(subs_M)


