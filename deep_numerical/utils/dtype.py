from   typing import Dict, Optional
import torch


__all__:           list[str]                     = ["dtype_real_to_complex", "dtype_complex_to_real", "type_as_real", "type_as_complex"]

__RTYPE_TO_CTYPE: Dict[torch.dtype, torch.dtype] = {getattr(torch, k): getattr(torch, f"c{k}") for k in ('float', 'double')}
__CTYPE_TO_RTYPE: Dict[torch.dtype, torch.dtype] = {_ctype: _rtype for (_rtype, _ctype) in __RTYPE_TO_CTYPE.items()}


##################################################
def dtype_real_to_complex(dtype: torch.dtype) -> torch.dtype:
    """Convert a real-valued dtype to its corresponding complex-valued dtype.

    ## Description
    Convert a real-valued data type to its corresponding complex-valued data type.

    ## Arguments
    `dtype` (`torch.dtype`): The real-valued data type.

    ## Returns
    `torch.dtype`: The corresponding complex-valued data type.
    """
    if dtype in __RTYPE_TO_CTYPE.keys():
        return __RTYPE_TO_CTYPE[dtype]
    else:
        raise TypeError(f"The dtype {dtype} is not supported.")


def dtype_complex_to_real(dtype: torch.dtype) -> torch.dtype:
    """Convert a complex-valued dtype to its corresponding real-valued dtype.

    ## Description
    Convert a complex-valued data type to its corresponding real-valued data type.

    ## Arguments
    `dtype` (`torch.dtype`): The complex-valued data type.

    ## Returns
    `torch.dtype`: The corresponding real-valued data type.
    """
    if dtype in __CTYPE_TO_RTYPE.keys():
        return __CTYPE_TO_RTYPE[dtype]
    else:
        raise TypeError(f"The dtype {dtype} is not supported.")


def type_as_real(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Get the corresponding real data type of a data type.

    ## Description
    Get the corresponding real data type of a given data type. If `dtype` is `None`, the default data type is used.

    ## Arguments
    `dtype` (`Optional[torch.dtype]`): The input data type. If `None`, the default data type is used.

    ## Returns
    `torch.dtype`: The corresponding real data type.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype in __RTYPE_TO_CTYPE.keys():
        return dtype
    elif dtype in __CTYPE_TO_RTYPE.keys():
        return __CTYPE_TO_RTYPE[dtype]
    else:
        raise TypeError(f"The dtype {dtype} is not supported.")


def type_as_complex(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Get the corresponding complex data type of a data type.

    ## Description
    Get the corresponding complex data type of a given data type. If `dtype` is `None`, the default data type is used.

    ## Arguments
    `dtype` (`Optional[torch.dtype]`): The input data type. If `None`, the default data type is used.

    ## Returns
    `torch.dtype`: The corresponding complex data type.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype in __CTYPE_TO_RTYPE.keys():
        return dtype
    elif dtype in __RTYPE_TO_CTYPE.keys():
        return __RTYPE_TO_CTYPE[dtype]
    else:
        raise TypeError(f"The dtype {dtype} is not supported.")


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()