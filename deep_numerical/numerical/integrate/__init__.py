from   typing      import Callable, Optional
import torch

from   .quadrature import *
from   .quadrature import __all__ as __all__quadrature


_FUNCTIONS: set[str]  = {'integration_guass_legendre', 'integration_gauss_legendre', 'integration_lebedev', 'integration_legendre', 'integration_S2'}
__all__:    list[str] = list(set(__all__quadrature) | _FUNCTIONS)


##################################################
def integration_guass_legendre(
    num_roots:   int,
    a:           float,
    b:           float,
    func:        Callable[..., torch.Tensor],
    func_kwargs: Optional[dict[str, object]] = None,
    dtype:       Optional[torch.dtype]       = None,
    device:      Optional[torch.device]      = None,
) -> torch.Tensor:
    """Numerical integration on a compact interval using the Gauss-Legendre quadrature rule.

    ## Description
    Computes numerical integration of a tensor-valued function `func` on `[a, b]` using the Gauss-Legendre quadrature rule of order `num_roots`.

    ## Arguments
    `num_roots` (`int`): The number of roots of the Gauss-Legendre quadrature rule.
    `a` (`float`): The lower bound of the interval.
    `b` (`float`): The upper bound of the interval.
    `func` (`Callable[..., torch.Tensor]`): The integrand function.
    `func_kwargs` (`Optional[dict[str, object]]`, default: `None`): Additional keyword arguments passed to `func`.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Data type of the roots and weights.
    `device` (`Optional[torch.device]`, default: `None`): Device for roots and weights.

    ## Returns
    `torch.Tensor`: The result of the numerical integration.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    if func_kwargs is None:
        func_kwargs = {}
    roots, weights = roots_legendre_shifted(num_roots, a, b, dtype=dtype, device=device)
    func_vals: torch.Tensor = func(roots, **func_kwargs)
    return torch.einsum("...t,t->...", func_vals, weights)


def integration_lebedev(
    f:                  Callable[[torch.Tensor], float],
    quad_order_lebedev: int                    = 7,
    dtype:              Optional[torch.dtype]  = None,
    device:             Optional[torch.device] = None,
) -> torch.Tensor:
    """Numerical integration on S2 using the Lebedev quadrature rule.

    ## Description
    Computes numerical integration of a function `f` over the unit 2-sphere $S^2$ using the Lebedev quadrature rule of order `quad_order_lebedev`.

    ## Arguments
    `f` (`Callable[[torch.Tensor], float]`): The integrand function.
    `quad_order_lebedev` (`int`, default: `7`): The order of the Lebedev quadrature rule.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Data type of the quadrature points.
    `device` (`Optional[torch.device]`, default: `None`): Device of the quadrature points.

    ## Returns
    `torch.Tensor`: The result of the spherical integration.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    roots, weights = roots_lebedev(quad_order_lebedev, dtype=dtype, device=device)
    return torch.sum(f(roots) * weights)


integration_gauss_legendre: Callable[..., torch.Tensor] = integration_guass_legendre
integration_legendre:       Callable[..., torch.Tensor] = integration_guass_legendre
integration_S2:             Callable[..., torch.Tensor] = integration_lebedev


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()