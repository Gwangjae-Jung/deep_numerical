from    typing      import  Sequence, Iterable, TypeAlias, TypeVar, Union
from    pathlib     import  Path
from    numpy       import  ndarray
import  torch


##################################################
##################################################
__all__ = [
    "SUPPORTED_ARRAY_MODULES",
    "SUPPORTED_ARRAY_TYPES",
    "ArrayData",
    "PathLike",
    "Objects",
    "TORCH_DEFAULT_DTYPE",
    "TORCH_DEFAULT_DEVICE",
    "dtype_real_to_complex",
    "dtype_complex_to_real",
]


##################################################
##################################################
SUPPORTED_ARRAY_MODULES = ("np", "torch")
SUPPORTED_ARRAY_TYPES   = (ndarray, torch.Tensor)
        

##################################################
##################################################
ArrayData:  TypeAlias   = Union[ndarray, torch.Tensor]
"""The typealias for the available tensors (`numpy.ndarray` and `torch.Tensor`)."""
PathLike:   TypeAlias   = Union[str, Path]
"""The typealias for the available path-like objects (`str` and `pathlib.Path`)."""

_T = TypeVar("_T")
Objects:    TypeAlias   = Union[_T, Sequence[_T], Iterable[_T]]
"""The typealias for the available objects (`Any`, `Sequence`, and `Iterable`)."""

TORCH_DEFAULT_DTYPE:    torch.dtype     = torch.float64
"""The default `torch` datatype for numerical opearations. It is set to `torch.float64` by default."""
TORCH_DEFAULT_DEVICE:   torch.device    = torch.device("cpu")
"""The default `torch` device for numerical opearations. It is set to `torch.device("cpu")` by default."""


def dtype_real_to_complex(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype is torch.float16:
        return torch.complex32
    elif real_dtype is torch.float32:
        return torch.complex64
    elif real_dtype is torch.float64:
        return torch.complex128
    else:
        raise TypeError(f"Input datatype: {real_dtype}")
    
    
def dtype_complex_to_real(complex_dtype: torch.dtype) -> torch.dtype:
    if complex_dtype is torch.complex32:
        return torch.float16
    elif complex_dtype is torch.complex64:
        return torch.complex32
    elif complex_dtype is torch.complex128:
        return torch.complex64
    else:
        raise TypeError(f"Input datatype: {complex_dtype}")


##################################################
##################################################
# End of file