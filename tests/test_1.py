import numpy as np
import scipy.sparse as sp
from pyMIQP import MIQP
np.random.seed(1)

""" Synthetic (sparse) integer least squares problem
    Unconstrained!
"""

M, N = 1000, 10
v_range = 100

true_x = np.random.randint(v_range, size=N)
noisy_x = true_x + np.random.normal(size=N) * v_range * 0.05
A = sp.random(M, N, density=0.5)
b = A * noisy_x

print('true_x: ', true_x)
print('noisy_x: ', noisy_x)

""" Solve with bonmin """
Q = A.T * A
c = -A.T * b

xlb = np.zeros(N)
xub = np.full(N, np.inf)
var_types = np.full(N, 2)

miqp = MIQP()
miqp.set_c(c)
miqp.set_Q(Q)
# A, glb, gub not set!
miqp.set_xlb(xlb)
miqp.set_xub(xub)
miqp.set_var_types(var_types)
miqp.solve(algorithm="B-Hyb")              # redundant: algorithm-default: B-Hyb
# miqp.solve(algorithm="BB")

print('sol-status: ', miqp.get_sol_status())
print('sol-obj: ', miqp.get_sol_obj())
print('sol-x: ', miqp.get_sol_x())
print('sol-time (ms): ', miqp.get_sol_time())
