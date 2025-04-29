from    typing      import  Sequence, Iterable, TypeAlias, TypeVar, Union
from    pathlib     import  Path
from    numpy       import  ndarray
from    torch       import  Tensor


##################################################
##################################################
__all__ = [
    "SUPPORTED_ARRAY_MODULES",
    "SUPPORTED_ARRAY_TYPES",
    "ArrayData",
    "PathLike",
    "Objects",
]


##################################################
##################################################
SUPPORTED_ARRAY_MODULES = ("np", "torch")
SUPPORTED_ARRAY_TYPES   = (ndarray, Tensor)
        

##################################################
##################################################
ArrayData:  TypeAlias   = Union[ndarray, Tensor]
"""The typealias for the available tensors (`numpy.ndarray` and `torch.Tensor`)."""
PathLike:   TypeAlias   = Union[str, Path]
"""The typealias for the available path-like objects (`str` and `pathlib.Path`)."""

_T = TypeVar("_T")
Objects:    TypeAlias   = Union[_T, Sequence[_T], Iterable[_T]]
"""The typealias for the available objects (`Any`, `Sequence`, and `Iterable`)."""


##################################################
##################################################
# End of file