from   typing            import Any, Callable, Union
from   typing_extensions import TypeAlias
import torch
from   torch.func        import jacfwd, jacrev, vmap


__all__:           list[str] = ['jacobian', 'hessian', 'derivatives']
FuncType:        TypeAlias = Callable[[torch.Tensor, Any], torch.Tensor]
FuncTypeWithAux: TypeAlias = Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]]


##################################################
def jacobian(func: FuncType, return_out: bool = False) -> Union[FuncType, FuncTypeWithAux]:
    """Computes the Jacobian function.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) which computes the Jacobian of `func` using `torch.func.jacrev`.
    If `return_out` is `True`, the returned function returns both the Jacobian and the function output.

    ## Arguments
    `func` (`FuncType`): A function mapping a tensor of points to output vectors.
    `return_out` (`bool`, default: `False`): Whether to also return the evaluated function output.

    ## Returns
    `Union[FuncType, FuncTypeWithAux]`: Vectorized function computing the Jacobian.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            out: torch.Tensor = func(pts, **kwargs)
            return out, out
        return vmap(jacrev(modified_func, has_aux=True))
    else:
        return vmap(jacrev(func))


def hessian(func: FuncType, return_out: bool = False) -> Union[FuncType, FuncTypeWithAux]:
    """Computes the Hessian function.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) which computes the Hessian of `func` using forward-over-reverse mode autodiff (`torch.func.jacfwd(torch.func.jacrev(...))`).
    If `return_out` is `True`, the returned function returns both the Hessian and the function output.

    ## Arguments
    `func` (`FuncType`): A function mapping a tensor of points to output vectors.
    `return_out` (`bool`, default: `False`): Whether to also return the evaluated function output.

    ## Returns
    `Union[FuncType, FuncTypeWithAux]`: Vectorized function computing the Hessian.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            out: torch.Tensor = func(pts, **kwargs)
            return out, out
        return vmap(jacfwd(jacrev(modified_func, has_aux=True), has_aux=True))
    else:
        return vmap(jacfwd(jacrev(func)))


def derivatives(func: FuncType, degree: int = 1) -> Union[FuncType, FuncTypeWithAux]:
    """Computes higher-order derivatives of a function.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) which iteratively computes derivatives of `func` up to the given `degree` using `torch.func.jacfwd`.
    If `return_out` is `True`, the returned function returns both the derivative and intermediate outputs.

    ## Arguments
    `func` (`FuncType`): A function mapping a tensor of points to output vectors.
    `degree` (`int`, default: `1`): The order/degree of differentiation (e.g., 1 for Jacobian, 2 for Hessian).
    `return_out` (`bool`, default: `False`): Whether to also return intermediate evaluation outputs.

    ## Returns
    `Union[FuncType, FuncTypeWithAux]`: Vectorized function computing the derivatives of order `degree`.
    """
    def modified_func(pts: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, list[Any]]:
        return func(pts, **kwargs), []

    def repack(_func: Any, _return_out: bool = True) -> Callable[..., Any]:
        def _repacked_func(pts: torch.Tensor, **kwargs: Any) -> Union[tuple[torch.Tensor, list[Any]], list[Any]]:
            _out, _aux = _func(pts, **kwargs)
            _aux.append(_out)
            if _return_out:
                return _out, _aux
            else:
                return _aux
        return _repacked_func

    _vmapped: Any = modified_func
    for _ in range(1, 1 + degree):
        _vmapped = jacfwd(repack(_vmapped), has_aux=True)
    return vmap(repack(_vmapped, False))


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()