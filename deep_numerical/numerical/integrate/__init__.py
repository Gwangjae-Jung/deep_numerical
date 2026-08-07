import  torch
from    typing          import  Callable, Optional
from    .quadrature     import  *
from    .quadrature     import  __all__ as  __all__quadrature


_FUNCTIONS = {'integration_guass_legendre', 'integration_lebedev', 'integration_legendre', 'integration_S2'}
__all__: list[str] = list(set(__all__quadrature) | _FUNCTIONS)


##################################################
##################################################
def integration_guass_legendre(
        num_roots:      int,
        a:              float,
        b:              float,
        func:           Callable[[torch.Tensor, object], torch.Tensor],
        func_kwargs:    Optional[dict[str, object]] = {},
        dtype:          Optional[torch.dtype]   = None,
        device:         Optional[torch.device]  = None,
    ) -> torch.Tensor:
    """
    ## Numerical integration on a compact interval using the Gauss-Legendre quadrature rule
    
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the Gauss-Legendre quadrature rule of order `num_roots`.
    
    Arguments:
        `num_roots` (`int`): The number of roots of the Gauss-Legendre quadrature rule.
        `a` (`float`): The lower bound of the interval.
        `b` (`float`): The upper bound of the interval.
        `func` (`Callable[[torch.Tensor, object], torch.Tensor]`):  The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `func_kwargs` (`Optional[dict[str, object]]`): The other arguments of `func`.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    roots, weights = roots_legendre_shifted(num_roots, a, b, dtype=dtype, device=device)
    func_vals = func(roots, **func_kwargs)
    return torch.einsum("...t,t->...", func_vals, weights)


def integration_lebedev(
        f:                  Callable[[torch.Tensor], float],
        quad_order_lebedev: int                 = 7,
        dtype:          Optional[torch.dtype]   = None,
        device:         Optional[torch.device]  = None,
    ) -> torch.Tensor:
    """
    ## Numerical integration on S2 using the Lebedev quadrature rule
    
    This function supports the numerical integration of a complex-valued function `f`, using the Lebedev quadrature rule of order `quad_order_lebedev`.
    
    Currently, the parallel computation is not implemented, which is easy to be implemented.
    
    Arguments:
        `f` (`Callable[[torch.Tensor], float]`): The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `quad_order_lebedev` (`int`): The order of the Lebedev quadrature rule.
        `dtype` (`torch.dtype`): The data type of the input tensor.
        `device` (`torch.device`): The device on which the input tensor is located.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    roots, weights = roots_lebedev(quad_order_lebedev, dtype=dtype, device=device)
    return torch.sum(f(roots) * weights)


integration_legendre    = integration_guass_legendre
integration_S2          = integration_lebedev


##################################################
##################################################
# End of file