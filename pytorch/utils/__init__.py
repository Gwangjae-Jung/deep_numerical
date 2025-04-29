"""## The utility module for various purposes.

-----
### Description
This module provides various features for a variety of implementation.
In this module, the following features are provided:
* `_autograd`
    provides several functions for the autograd operations. Two main functions are `compute_grad` and `compute_jacobian`.
* `_constants`
    provides several constants which are used throughout this package.
* `_dtype` (copied from `utils_main`)
    provides several type hints which are used throughout this package.
* `_fft_utils`
    provides several helper functions for the FFT operations and its application (for example, efficient convolutions).
* `_grid`
    provides several functions which generates the uniform regular grid on a given box.
* `_helper` (copied from `utils_main`)
    provides several helper functions which are used throughout this package.
* `_math`
    provides several frequently used math functions.
* `_num_int`
    provides several functions for numerical integration.
* `_quadrature`
    provides several quadrature rules, which can be easily applied in practice.
    It also provides several grids on compact boxes, disks, and spheres.
* `_runge_kutta`
    provides several Runge-Kutta methods for a single march.
"""
from    ._dtype             import  *
from    ._helper            import  *

from    ._constants         import  *
from    ._fft_utils         import  *
from    ._math              import  *
from    ._num_int           import  *
from    ._quadrature        import  *
from    ._runge_kutta       import  *
from    ._grid              import  *


from    ._autograd      import  *
from    ._network       import  *


from    ._uncategorized     import  *


##################################################
##################################################
# End of file