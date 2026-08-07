import  torch
from    typing      import  Any, Callable, Union, TypeAlias
from    torch.func  import  vmap, jacfwd, jacrev


__all__ = ['jacobian', 'hessian', 'derivatives',]


##################################################
##################################################
FuncType: TypeAlias = Callable[[torch.Tensor, Any], torch.Tensor]
FuncTypeWithAux: TypeAlias = Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]]


def jacobian(func: FuncType, return_out: bool=False) -> Union[FuncType, FuncTypeWithAux]:
    """Returns a function which computes the Jacobian of `func`.
    
    Arguments:
        `func` (`FuncType`): A Python function which maps a `k`-dimensional vector to a `d`-dimensional vector.
        `return_out` (`bool`, default: `False`): Whether the output should also be given.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
            out = func(pts, **kwargs)
            return out, out
        return vmap(jacrev(modified_func, has_aux=True))
    else:
        return vmap(jacrev(func))


def hessian(func: FuncType, return_out: bool=False) -> Union[FuncType, FuncTypeWithAux]:
    """Returns a function which computes the Hessian of `func`.
    
    Arguments:
        `func` (`FuncType`): A Python function which maps a `k`-dimensional vector to a `d`-dimensional vector.
        `return_out` (`bool`, default: `False`): Whether the output should also be given.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
            out = func(pts, **kwargs)
            return out, out
        return vmap(jacfwd(jacrev(modified_func, has_aux=True), has_aux=True))
    else:
        return vmap(jacfwd(jacrev(func)))


def derivatives(func: FuncType, degree: int=1) -> Union[FuncType, FuncTypeWithAux]:
    """Returns a function which computes the derivatives of `func` of order `degree`.
    
    Arguments:
        `func` (`FuncType`): A Python function which maps a `k`-dimensional vector to a `d`-dimensional vector.
        `degree` (`int`, default: `1`): The degree of differentiation. (Example: `1` for Jacobian, `2` for Hessian)
        `return_out` (`bool`, default: `False`): Whether the output should also be given.
    """
    # Define helper functions
    def modified_func(pts: torch.Tensor, **kwargs) -> FuncTypeWithAux:
        return func(pts, **kwargs), []
    def repack(_func: FuncTypeWithAux, _return_out: bool=True) -> FuncTypeWithAux:
        def _repacked_func(pts: torch.Tensor, **kwargs) -> list[torch.Tensor]:
            _out, _aux = _func(pts, **kwargs)
            _aux.append(_out)
            if _return_out: return _out, _aux
            else:           return _aux
        return _repacked_func
    # Compute derivatives
    _vmapped = modified_func
    for _ in range(1, 1+degree):
        _vmapped = jacfwd(repack(_vmapped), has_aux=True)
    return vmap(repack(_vmapped, False))


##################################################
##################################################
# End of file