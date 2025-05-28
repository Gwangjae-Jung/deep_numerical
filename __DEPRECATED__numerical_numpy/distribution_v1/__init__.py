"""## The module for computation of several distribution functions and their physical quantities

-----
### Warning
This module is the previous version of the module `distribution`, and this module partially supports inhomogeneous distribution functions.

-----
### Description
This module provides some functions which are used to compute several distribution functions and their physical quantities.
    * `_density_function`
        provides several density functions (distribution functions), which can be used as test functions.
    * `_physical_quantity`
        provides functions for the computation of some physical quantities; density, momentum, energy, entropy, etc.
-----
### Note

1. Notation
    Throughout this submodule, we define the following notaions.
        * `B`: The number of the instances,
        * `d`: The dimension of the space.
        * `(N_1, ..., N_d)`: The shape of the spatial grid. Hence, the shape of the spatial grid with the coordinates is `(N_1, ..., N_d, d)`.
        * `(K_1, ..., K_d)`: The shape of the velocity grid. Hence, the shape of the velocity grid with the coordinates is `(K_1, ..., K_d, d)`.

2. Shapes of the input tensors
    We assume that all input tensors for the distribution functions are of the following shape:
        `(num_instances, *physical_domain, *velocity_space, num_functions)`.
"""
from    ._density_function      import  *
from    ._physical_quantity     import  *


##################################################
##################################################
# End of file