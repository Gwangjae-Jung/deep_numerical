"""## Neural operators - Neural networks for approximating continuous operators

-----
### Description
This submodule provides some classes of neural operators which can be used to approximate continuous operators.

-----
### Features

* Deep Operator Network (DeepONet) and Multiple-Input Operator Network (MIONet)
* Graph Neural Operator (GNO)
* Fourier Neural Operator (FNO) and its variants (FFNO, SFNO, TFNO)
* Galerkin Transformer (GT)
* (Under construction) Multipole Graph Neural Operator (MGNO)
"""
from    .fno    import  *
from    .ffno   import  *
from    .sfno   import  *
from    .tfno   import  *
from    .onet   import  *
from    .gt     import  *
from    .gno    import  *

from    .hyper_onet     import  *
from    .hyper_sfno     import  *


##################################################
##################################################
# End of file