"""# Python library for numerical methods for solving kinetic equations with neural network architectures

-----
This library provides spectral methods for solving kinetic equations and a collection of neural network architectures.

## A. Numerical methods
The numerical methods provided by this library can be found in the submodule `deep_numerical.numerical`, which includes spectral methods for kinetic equations:
1. Classical spectral method for the Boltzmann equation
    * Only the solver for the elastic Boltzmann equation is implemented.
2. Fast spectral method
    * Fokker-Planck-Landau equation
    * Boltzmann equation with general collision kernels

## B. Neural network architectures
The neural network architectures provided by this library can be found in the submodule `deep_numerical.neural`.

-----
## References

[1] G. Dimarco and L. Pareschi, Numerical methods for kinetic equations, Acta Numer., 23 (2014), pp. 369–520, https://doi.org/10.1017/S0962492914000063.

[2] I. M. Gamba, J. R. Haack, C. D. Hauck, and J. Hu, A fast spectral method for the Boltzmann collision operator with general collision kernels, SIAM J. Sci. Comput., 39 (2017), pp. B658–B674, https://doi.org/10.1137/16M1096001.
"""
from    typing              import  TYPE_CHECKING
from    typing              import  TypeVar, Generic, Iterable, Union, Set
from    typing_extensions   import  Self, TypeAlias
import  importlib


if TYPE_CHECKING:
    from    numpy   import  ndarray
    from    torch   import  Tensor
    from    .   import  autograd
    from    .   import  fft
    from    .   import  neural
    from    .   import  numerical
    from    .   import  utils

    ArrayData:  TypeAlias   = Union[ndarray, Tensor]
    """The typealias for the available tensors (`numpy.ndarray` and `torch.Tensor`)."""


_SUBMODULES:    Set[str] = {'autograd', 'fft', 'neural', 'numerical', 'utils'}
_VARIABLES:     Set[str] = {'Objects', 'ArrayData', 'EINSUM_STRING'}
_FUNCTIONS:     Set[str] = {'repeat', 'ones', 'zeros'}
__all__ = list(_SUBMODULES | _VARIABLES | _FUNCTIONS)


##################################################
T = TypeVar("T")
Objects:    TypeAlias   = Union[T, Iterable[T]]
"""The typealias for the available objects (`Any` and `Iterable`)."""


EINSUM_STRING:  str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""The string of 26 uppercase alphabets (`ABC...XYZ`), which is used to define an Einstein summation command by slicing the string."""


class repeat(Generic[T]):
    """## Repeat iterator
    
    ## Description
    An iterator that yields the same object `k` times.
    """
    def __init__(self, obj: T, k: int) -> None:
        """## The initializer of `repeat`
        
        ## Description
        Initializes the iterator with an object and a repetition count.
        
        ## Arguments
        `obj` (`T`): The object to repeat.
        `k` (`int`): The number of times to yield `obj`.
        
        ## Returns
        `None`: None.
        """
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


def ones(k: int) -> repeat[int]:
    """## Iterator repeating integer 1
    
    ## Description
    Returns a `repeat` iterator that yields integer `1` for `k` times.
    
    ## Arguments
    `k` (`int`): The number of repetitions.
    
    ## Returns
    `repeat[int]`: An iterator yielding 1 `k` times.
    """
    return repeat(1, k)


def zeros(k: int) -> repeat[int]:
    """## Iterator repeating integer 0
    
    ## Description
    Returns a `repeat` iterator that yields integer `0` for `k` times.
    
    ## Arguments
    `k` (`int`): The number of repetitions.
    
    ## Returns
    `repeat[int]`: An iterator yielding 0 `k` times.
    """
    return repeat(0, k)


##################################################
def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    elif name == 'ArrayData':
        from numpy import ndarray
        from torch import Tensor
        val = Union[ndarray, Tensor]
        globals()['ArrayData'] = val
        return val
    elif name == 'Tensor':
        from torch import Tensor
        globals()['Tensor'] = Tensor
        return Tensor
    elif name == 'ndarray':
        from numpy import ndarray
        globals()['ndarray'] = ndarray
        return ndarray
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'deep_numerical' has no attribute '{name}'.")


##################################################
if __name__ == '__main__':
    import os, sys
    sys.path.append(".")
    os.system("cls")
    print("Begin import...")
    import autograd
    print(f"autograd: {autograd.__name__}")
    import fft
    print(f"fft: {fft.__name__}")
    import neural
    print(f"neural: {neural.__name__}")
    import numerical
    print(f"numerical: {numerical.__name__}")

    from neural.layer import MLP
    mlp = MLP([3, 10, 2])
    print(mlp.forward(X=11))
    

##################################################
# End of file