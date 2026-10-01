"""## Implementation of spectral numerical methods for solving kinetic equations

-----
### Description
This module provides several classes which compute the numerical solution of several kinetic equations based on the spectral method (Fourier-Galerkin method).

* `dsm`
    provides the direct spectral method for solving the Boltzmann equation.
* `fsm__boltzmann`
    provides the fast spectral method which can be used for general collision kernels, suggested in [(Gamba, 2017)](https://epubs.siam.org/doi/10.1137/16M1096001).
* `fsm__fpl`
    provides the fast spectral method for the Fokker-Planck-Landau equation, with the fast algorithm suggested in [(Pareschi, 2000)](https://www.sciencedirect.com/science/article/pii/S0021999100966129).

-----
### Reference
[(Gamba, 2017)]: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)

[(Pareschi, 2000)]: [L. Pareschi, G. Russo, G. Toscani, Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator, Journal of Computational Physics, Volume 165, Issue 1, 2000, Pages 216-236](https://www.sciencedirect.com/science/article/pii/S0021999100966129).

-----
### Note
All implementation of numerical methods assumes that both the input and output tensors (which are instantaneous records) are of the following shape: `(num_batch, *physical_domain, *velocity_domain, num_functions)`.
When stacked to form the resultant datasets, the instantaneous data are stacked along `axis=1`, forming a tensor of shape `(num_batch, num_timestamps, *physical_domain, *velocity_domain, num_functions)`.
"""
# # Spectral method
from    .base_classes       import  *
from    .constants          import  *
from    .dsm                import  *
from    .fsm__boltzmann     import  *
from    .fsm__fpl           import  *

from    .runge_kutta        import  *

from    .base_classes       import  __all__ as  __all__base_classes
from    .constants          import  __all__ as  __all__constants
from    .dsm                import  __all__ as  __all__dsm
from    .fsm__boltzmann     import  __all__ as  __all__fsm__boltzmann
from    .fsm__fpl           import  __all__ as  __all__fsm__fpl
from    .runge_kutta        import  __all__ as  __all__runge_kutta


__all__: list[str] = (
    __all__base_classes
    + __all__constants
    + __all__dsm
    + __all__fsm__boltzmann
    + __all__fsm__fpl
    + __all__runge_kutta
)


##################################################
##################################################
# End of file