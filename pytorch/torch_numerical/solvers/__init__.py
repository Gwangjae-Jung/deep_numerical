"""## Implementation of some numerical methods for solving kinetic equations

-----
### Description
This module provides several classes which computes the numerical solution of several kinetic equations.
Currently, both the discrete velocity method and the spectral method (based on the Fourier-Galerkin method) are supported.

1. Discrete velocity method (DVM)
    * `_dvm__bgk`
        provides the numerical solution of the BGK model.
    * `_dvm_sampling`
        provides the numerical solution of the Boltzmann equation, based on the sampling method suggested by [(Goldstein, 1989)](https://arc.aiaa.org/doi/10.2514/5.9781600865923.0100.0117).

2. Spectral methods (DSM and FSM)
    * `_dsm`
        provides the direct spectral method for solving the Boltzmann equation.
    * `_fsm_general`
        provides the fast spectral method which can be used for any collision kernel, suggested in [(Gamba, 2017)](https://epubs.siam.org/doi/10.1137/16M1096001).
    * `_fsm__fpl`
        provides the fast spectral method for the Fokker-Planck-Landau equation, with the fast algorithm suggested in [(Pareschi, 2000)](https://www.sciencedirect.com/science/article/pii/S0021999100966129).

-----
### Reference
[(Goldstein, 1989)]: [D. Goldstein, B. Sturtevant and J. Broadwell (1989), Investigation of the motion of discrete velocity gases, Technical Report 118, Rar. Gas. Dynam., Progress in Astronautics e Aeronautics, AIAA, Washington](https://arc.aiaa.org/doi/10.2514/5.9781600865923.0100.0117)

[(Gamba, 2017)]: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)

[(Pareschi, 2000)]: [L. Pareschi, G. Russo, G. Toscani, Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator, Journal of Computational Physics, Volume 165, Issue 1, 2000, Pages 216-236](https://www.sciencedirect.com/science/article/pii/S0021999100966129).

-----
### Note
All implementation of numerical methods assumes that both the input and output tensors (which are instantaneous records) are of the following shape: `(num_batch, *physical_domain, *velocity_domain, num_functions)`.
When stacked to form the resultant datasets, the instantaneuous data are stacked along `axis=1`, forming a tensor of shape `(num_batch, num_timestamps, *physical_domain, *velocity_domain, num_functions)`.`
"""
# Discrete velocity method
from    ._dvm__bgk          import  *
from    ._dvm_sampling      import  *

# Spectral method
from    ._dsm               import  *
from    ._fsm__boltzmann    import  *
from    ._fsm__fpl          import  *


##################################################
##################################################
# End of file