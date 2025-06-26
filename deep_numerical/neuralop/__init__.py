"""## Neural operators - Neural networks for approximating continuous operators

-----
### Description
This submodule provides some classes of neural operators which can be used to approximate continuous operators.

-----
### Features

* Deep Operator Network (DeepONet) and Multiple-Input Operator Network (MIONet)
* Graph Neural Operator (GNO)
* Fourier Neural Operator (FNO) and its variants (Factorized FNO, Separable FNO, Tensorized FNO, Radial FNO)
* Galerkin Transformer (GT)
* (Under construction) Multipole Graph Neural Operator (MGNO)
"""
from    .fno                import  *
from    .fno_factorized     import  *
from    .fno_separable      import  *
from    .fno_tensorized     import  *
from    .fno_radial         import  *

from    .onet   import  *
from    .gt     import  *
from    .gno    import  *

from    .hyper_onet     import  *
from    .hyper_sfno     import  *


##################################################
##################################################
# End of file