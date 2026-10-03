"""Implementation of spectral numerical methods for solving kinetic equations.

-----
### Description
This module provides classes which compute the numerical solution of several kinetic equations based on the spectral method (Fourier-Galerkin method):
* `dsm`: Direct spectral method for the Boltzmann equation.
* `fsm__boltzmann`: Fast spectral method for general collision kernels.
* `fsm__fpl`: Fast spectral method for the Fokker-Planck-Landau equation.

-----
### References
[1] I. M. Gamba, J. R. Haack, C. D. Hauck, and J. Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM J. Sci. Comput., 39 (2017), pp. B658–B674.
[2] L. Pareschi, G. Russo, and G. Toscani, Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator, J. Comput. Phys., 165 (2000), pp. 216–236.
"""
from .base_classes   import *
from .constants      import *
from .dsm            import *
from .fsm__boltzmann import *
from .fsm__fpl       import *
from .runge_kutta    import *

from .base_classes   import __all__ as __all__base_classes
from .constants      import __all__ as __all__constants
from .dsm            import __all__ as __all__dsm
from .fsm__boltzmann import __all__ as __all__fsm__boltzmann
from .fsm__fpl       import __all__ as __all__fsm__fpl
from .runge_kutta    import __all__ as __all__runge_kutta


__all__: list[str] = (
    __all__base_classes
    + __all__constants
    + __all__dsm
    + __all__fsm__boltzmann
    + __all__fsm__fpl
    + __all__runge_kutta
)


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()