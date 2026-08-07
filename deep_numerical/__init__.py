"""# Python library for numerical methods for solving kinetic equations with neural network architectures

-----
This library provides spectral methods for solving kinetic equations and a collection of neural network architectures.

## A. Numerical methods
The numerical methods provided by this library can be found in the submodule `deep_numerical.numerical`, which includes the following methods.
1. Discrete velocity method (DVM): To be constructed.
    1. Classical DVM
    2. Fast DVM
2. Spectral method
    1. Classical spectral method for the Boltzmann equation
        1. Only the solver for the elastic Boltzmann equation is implemented.
    2. Fast spectral method
        1. (Fokker-Planck-Landau equation) Only the solver for the elastic FPL equation is implemented.
        2. (Boltzmann equation) Reference should be given.

## B. Neural network architectures
The neural network architectures provided by this library can be found in the submodules `deep_numerical.nn` and `deep_numerical.neuralop`.

-----
## References

[1] G. Dimarco and L. Pareschi, Numerical methods for kinetic equations, Acta Numer., 23 (2014), pp. 369–520, https://doi.org/10.1017/S0962492914000063.

[2] I. M. Gamba, J. R. Haack, C. D. Hauck, and J. Hu, A fast spectral method for the Boltzmann collision operator with general collision kernels, SIAM J. Sci. Comput., 39 (2017), pp. B658–B674, https://doi.org/10.1137/16M1096001.
"""
from    typing      import  TYPE_CHECKING
from    typing      import  TypeAlias, TypeVar, Generic, Iterable, Union, Set, Self
import  importlib
from    numpy       import  ndarray
from    torch       import  Tensor


if TYPE_CHECKING:
    from    .   import  autograd
    from    .   import  fft
    from    .   import  neural
    from    .   import  numerical


_SUBMODULES:    Set = {'autograd', 'fft', 'neural', 'numerical'}
_VARIABLES:     Set = {'Objects', 'ArrayData', 'EINSUM_STRING'}
_FUNCTIONS:     Set = {'repeat', 'ones', 'zeros'}
__all__ = list(_SUBMODULES | _VARIABLES | _FUNCTIONS)


##################################################
##################################################
T = TypeVar("T")
Objects:    TypeAlias   = Union[T, Iterable[T]]
"""The typealias for the available objects (`Any` and `Iterable`)."""
ArrayData:  TypeAlias   = Union[ndarray, Tensor]
"""The typealias for the available tensors (`numpy.ndarray` and `torch.Tensor`)."""


EINSUM_STRING:  str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""The string of 26 uppercase alphabets (`ABC...XYZ`), which is used to define an Einstein summation command by slicing the string."""


class repeat(Generic[T]):
    def __init__(self, obj: T, k: int) -> None:
        self.__object = obj
        self.__k = k
        self.__current = 0
        return None
    
    def __iter__(self) -> Self:
        return self
    
    def __next__(self) -> T:
        if self.__current < self.__k:
            self.__current += 1
            return self.__object
        else:
            raise StopIteration


def ones(k: int) -> repeat[int]:    return repeat(1, k)
def zeros(k: int) -> repeat[int]:   return repeat(0, k)


##################################################
##################################################
def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'deep_numerical' has no attribute '{name}'.")


##################################################
##################################################
# End of file