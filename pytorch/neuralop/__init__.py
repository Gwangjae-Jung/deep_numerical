"""## Neural operators - Neural networks for approximating continuous operators

-----
### Description
This submodule provides some classes of neural operators which can be used to approximate continuous operators.

-----
### Features

* Deep Operator Network and Multiple-Input Operator Network
* Graph Neural Operator
* Fourier Neural Operator
* Galerkin Transformer
* (UNSTABLE) Multipole Graph Neural Operator
"""
from    .fno    import  *
from    .ffno   import  *
from    .onet   import  *
from    .gt     import  *
from    .gnot   import  *
from    .gno    import  *

from    .hyper_onet     import  *


##################################################
##################################################
# End of file