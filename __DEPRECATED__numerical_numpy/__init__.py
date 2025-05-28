"""# (`Deprecated`) A package for numerical methods for solving kinetic equations

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
import  warnings
warnings.warn(
    'This package will no longer be maintained, and any reported errors raised in this package will not be fixed. Please use the new package `python_deep_numerical.pytorch.numerical` instead.',
    DeprecationWarning
)

from    .   import  distribution
from    .   import  solvers
from    .   import  utils

# Include version 1
from    .   import  distribution_v1
from    .   import  solvers_v1


##################################################
##################################################
# End of file