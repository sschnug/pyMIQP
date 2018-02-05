import numpy as np
import scipy.sparse as sp
from pyMIQP import MIQP

"""
Example from OPTI:
    https://www.inverseproblem.co.nz/OPTI/index.php/Probs/MIQP
    Example 1: Small MIQP
"""

Q = sp.csc_matrix(np.array([[1,-1],[-1,2]]))
c = np.array([-2, -6])

A = sp.csc_matrix(np.array([[1,1],[-1,2], [2,1]]))
glb = np.array([-np.inf, -np.inf, -np.inf])
gub = np.array([2,2,3])

xlb = np.array([0,0])
xub = np.array([np.inf, np.inf])

# 0, 1, 2 = cont, bin, int
var_types = np.array([2, 0])

miqp = MIQP()
miqp.set_c(c)
miqp.set_Q(Q)
miqp.set_A(A)
miqp.set_glb(glb)
miqp.set_gub(gub)
miqp.set_xlb(xlb)
miqp.set_xub(xub)
miqp.set_var_types(var_types)
miqp.set_initial_point(np.array([0, 0]))  # redundant
miqp.solve_bb()

print('sol-status: ', miqp.get_sol_status())
print('sol-obj: ', miqp.get_sol_obj())
print('sol-x: ', miqp.get_sol_x())
