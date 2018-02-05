# pyMIQP
This is a **Mixed Integer Quadratic Programming** solver for python.

Internally it uses [CoinOR](https://www.coin-or.org/)'s general MINLP-solver [bonmin](https://www.coin-or.org/Bonmin/) (which uses other CoinOR projects like [Cbc](https://projects.coin-or.org/Cbc) and [Ipopt](https://projects.coin-or.org/Ipopt))
and prepares the necessary internals tuned for instances of Quadratic Programming (automatic calculation
of NLP-like specification-components like gradients/jacobians).

In general, Bonmin tackles MINLP (Mixed Integer NonLinear Programming) problems
which is more general than MIQP (Mixed Integer Quadratic Programming) problems,
but the performance, when specialized commercial solvers (Gurobi, CPLEX, Mosek;
some potentially limited to CMIQP -> convex) are unavailable, is good (only
open-source + free solver considered in [this Benchmark](http://plato.asu.edu/ftp/convex.html))!  

For convex problems, Bonmin guarantees a global-optimum, while it only guarantees
a local-optimum for nonconvex problems.

# Status
*Prototype*

It's working for small constrained examples and bigger sparse (unconstrained) integer-least squares problems.
But heavy-testing has not been done yet and it's to be expected, that there are bugs.

I'm rediscovering C++ right now and the C++-part in this project is sub-optimal at least!

# Problem-definition
    min           c.T x + 0.5 * x.T Q x
    subject to:   glb <= A x <= gub
                   lb <=   x <=  ub

                  x_i in Z for all i in I and,
                  x_i in R for all i not in I.

    with:
          c:           n   real vector
          Q: symmetric n*n real matrix
             convex if Q is positive-semidefinite
          A:           m*n real matrix
          glb:         m   real vector
          gub:         m   real vector
           lb:         n   real vector
           ub:         n   real vector

From this follows, that only *linear-constraints* are supported (limited by this wrapper; not due to Bonmin)!

# Implementation Overview
This wrapper uses [pybind11](https://github.com/pybind/pybind11) to combine C++ and Python.

The setuptools-based install is based on [pybind/python_example](https://github.com/pybind/python_example).

### C++Side

[Eigen](http://eigen.tuxfamily.org) is used for most (sparse) calculations.

### Python-side
The implementation assumes the usage of *sparse-matrices* (for Q + A), provided by [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) (see tests) and *numpy-arrays* (for vectors) provided by [numpy](http://www.numpy.org/).

# (Some) Implementation details
Currently [Ipopt's](https://projects.coin-or.org/Ipopt) [Hessian-approximation](https://www.coin-or.org/Ipopt/documentation/node31.html) feature is hardcoded!

This is due to the fact, that the necessary Hessian-calculation is not implemented yet.
In general, not using Hessian-approximation will result in faster and more robust performance,
except for the case where the Hessian is dense!

# Usage
See *tests* folder. E.g. ```test_0.py```:

    import numpy as np
    import scipy.sparse as sp
    from pyMIQP import MIQP

    # Example from OPTI:
    # https://www.inverseproblem.co.nz/OPTI/index.php/Probs/MIQP
    # Example 1: Small MIQP

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
    miqp.set_initial_point(np.array([0, 0]))                       # redundant
    miqp.solve_bb()

    print('sol-status: ', miqp.get_sol_status())
    print('sol-obj: ', miqp.get_sol_obj())
    print('sol-x: ', miqp.get_sol_x())                             # numpy-array

# Install
**Only the combination of Linux (Ubuntu-like) and python3 was tested!**

Make sure to respect all those licenses (a lot of software involved).

## Install Bonmin (system-wide)
- Prepare your system to be able to build software (```sudo apt install build-essential``` and co.)
  - The following might suffice (no guarantees): ```sudo apt-get install gfortran gcc g++ autotools-dev automake autoconf```
- Grab a release (e.g. from [github](https://github.com/coin-or/Bonmin/releases))
  - *Release* because: we need the ThirdParty folder (not shipped with all kinds of source-grabbing approaches)!
  - We are approaching a free open-source build here (```Cbc``` shipped; ```Ipopt``` shipped; ```Mumps````used as Linear-solver; will be grabbed)
    - Other configs were not tested!
  - cd to ```ThirdParty/Blas```
  - ```./get.Blas```
  - cd to ```ThirdParty/Lapack```
  - ```./get.Lapack```
  - cd to ```ThirdParty/Mumps```
  - ```./get.Mumps```
  - cd to core-dir
  - ```./configure --prefix=/usr --enable-cbc-parallel --enable-gnu-packages```
    - (global-install is used so that bonmin's lib is automatically found)
  - ```make -j 4```
  - (```make test```)
  - ```sudo make install```

## Grab Eigen
- Eigen is a header-only library and it's expected that the folder can be found at: ```src/Eigen```
  - ```src/Eigen/src/Eigen``` is available

## Install project
- ```sudo python3 setup.py install```

# ToDo
- [ ] C++ style-formatting: very ugly mixed-style code for now
- [ ] C++ code quality improvements: partially non-nice C++ Code
- [ ] Support Bonmin's options (algorithms and co.)
- [ ] Support options to control verbosity
- [ ] Error-handling
- [ ] (maybe) consider supporting MIQCQP (quadratic constraints) or MISOCP (second-order cone constraints)
- [ ] Provide Hessian-calculations to not use Ipopt's Hessian-approximation  
- [ ] Write an interface for cvxpy's newly introduced QP-pipeline

# Alternatives
The only alternatives known to the author:

- [Pyomo](http://www.pyomo.org/)
  - Very complete and mature software with sparse documentation
  - Not limited to QPs
  - As far as the author knows: Bonmin-usage completely based on [AMPL Solver Library](https://ampl.com/resources/hooking-your-solver-to-ampl/) (file-based?)
  - Usage resembles mathematical-modelling languages
- [CasADi](https://github.com/casadi/casadi/wiki)
  - Very complete software
  - Not limited to QPs