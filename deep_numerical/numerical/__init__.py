"""A package for numerical methods for solving kinetic equations.

-----
### Description
This package provides modules which can be used to develop models which solve kinetic equations.
In this package, the following features are provided:
    * Underlying utility functions.
    * Numerical methods for solving kinetic equations, including the discrete velocity method and the spectral method (the Fourier-Galerkin method).

-----
### Note
1. When using this package, users should be aware that every input tensor is assumed to be a collection of multiple instances. Hence, even if a tensor of one instance should be given, it should be reshaped to have the shape of `(1, ...)`.
"""
import importlib
from   typing import Any, Set, TYPE_CHECKING

from   .math  import *
from   .math  import __all__ as __all__math


_SUBMODULES: Set[str]  = {'distribution', 'integrate', 'math', 'solver'}
_MATH:       Set[str]  = set(__all__math)
__all__:     list[str] = list(_SUBMODULES | _MATH)

if TYPE_CHECKING:
    from . import distribution
    from . import integrate
    from . import math
    from . import solver


##################################################
def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()