"""The module for computation of several distribution functions and their physical quantities.

-----
### Description
This module provides functions which are used to compute several distribution functions and their physical quantities:
    * `_density_function`: provides several density functions (distribution functions), which can be used as test functions.
    * `_physical_quantity`: provides functions for the computation of some physical quantities; density, momentum, energy, entropy, etc.
    * `_collision_kernels`: provides collision kernels.

-----
### Note
1. Notation
    Throughout this submodule, we define the following notations:
        * `B`: The number of instances,
        * `d`: The dimension of the space.
        * `(N_1, ..., N_d)`: The shape of the spatial grid.
        * `(K_1, ..., K_d)`: The shape of the velocity grid.

2. Shapes of the input tensors
    All input tensors for the distribution functions are of the shape:
        `(num_instances, *physical_domain, *velocity_space, num_functions)`.
"""
from deep_numerical.numerical.distribution._density_function  import *
from deep_numerical.numerical.distribution._physical_quantity import *
from deep_numerical.numerical.distribution._collision_kernels import *

from deep_numerical.numerical.distribution._density_function  import __all__ as __all_density
from deep_numerical.numerical.distribution._physical_quantity import __all__ as __all_physical
from deep_numerical.numerical.distribution._collision_kernels import __all__ as __all_collision


__all__: list[str] = list(__all_density) + list(__all_physical) + list(__all_collision)


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()