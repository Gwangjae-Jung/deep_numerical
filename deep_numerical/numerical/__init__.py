"""# A package for numerical methods for solving kinetic equations

-----
### Description
This package provides modules which can be used to develop models which solves kinetic equations.
In this package, the following features are provided:
    * Underlying utility functions.
    * Numerical methods for solving kinetic equations, including the discrete velocity method and the spectral method (the Fourier-Galerkin method).

-----
### Note
1. When using this package, users should be aware that every input tensor is assumed to be a collection of multiple instances. Hence, even if a tensor of one instance should be given, it should be reshaped to have the shape of `(1, ...)`.
"""
from    .   import  distribution
from    .   import  solvers
# from    .   import  utils


##################################################
##################################################
# End of file