"""## Python library for numerical methods for solving kinetic equations with neural network architectures

-----
This library provides a collection of numerical methods for solving kinetic equations, such as the Fokker-Planck-Landau equation and the Boltzmann equation. It also provides a collection of neural network architectures.

### A. Numerical methods
The numerical methods provided by this library can be found in the submodule `deep_numerical.numerical`, which includes the following methods:

1. Discrete velocity method (DVM): To be constructed.
    1. Classical DVM
    2. Fast DVM
2. Spectral method
    1. Classical spectral method for the Boltzmann equation
        1. Only the solver for the elastic Boltzmann equation is implemented.
    2. Fast spectral method
        1. (Fokker-Planck-Landau equation) Only the solver for the elastic FPL equation is implemented.
        2. (Boltzmann equation) Reference should be given.

### B. Neural network architectures
The neural network architectures provided by this library can be found in the submodules `deep_numerical.nn` and `deep_numerical.neuralop`.

-----
### References
[1] G. Dimarco and L. Pareschi, Numerical methods for kinetic equations, Acta Numer., 23 (2014), pp. 369–520, https://doi.org/10.1017/S0962492914000063.
[2] I. M. Gamba, J. R. Haack, C. D. Hauck, and J. Hu, A fast spectral method for the Boltzmann collision operator with general collision kernels, SIAM J. Sci. Comput., 39 (2017), pp. B658–B674, https://doi.org/10.1137/16M1096001.
"""

from    .   import  utils
from    .   import  layers
from    .   import  nn
from    .   import  neuralop
from    .   import  numerical


##################################################
##################################################
# End of file