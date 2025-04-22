from    typing          import  Optional, Sequence, Union
import  torch
from    scipy.special   import  gamma


##################################################
##################################################
__all__: list[str] = [
    'sinc',
    'phase',
    'area_of_unit_sphere',
    'volume_of_unit_ball',
    'absolute_error',
    'relative_error',
]


##################################################
##################################################
def sinc(x: torch.Tensor) -> torch.Tensor:
    """Returns `sin(x)/x`."""
    return torch.where(
        x!=0,
        torch.sin(x)/x,
        torch.ones_like(x, dtype=x.dtype, device=x.device)
    )


def phase(theta: torch.Tensor) -> torch.Tensor:
    """Returns `exp(i * theta)`."""
    return torch.exp(1j * theta)


##################################################
##################################################
def area_of_unit_sphere(dim_embed: int) -> float:
    r"""Returns the area of the unit sphere $S^{d-1}$ embeded in $\mathbb{R}^d$."""
    return 2 * (torch.pi**(dim_embed/2)) / gamma(dim_embed/2)


def volume_of_unit_ball(dim_embed: int) -> float:
    r"""Returns the volume of the unit ball embeded in $\mathbb{R}^d$."""
    return torch.pi**(dim_embed/2) / gamma(1 + dim_embed/2)


##################################################
##################################################
def absolute_error(
        preds:      torch.Tensor,
        targets:    torch.Tensor,
        p:          Union[float, str] = 2.0,
        dim:        Optional[Sequence[int]] = None,
        scale:      Optional[torch.Tensor]  = None,
    ) -> torch.Tensor:
    """Returns the instance-wise absolute error between the `preds` and `targets`.
    
    -----
    Given to sequences `preds` and `targets` of shape `(N, ...)` (where `N` is the number of the instances), this function returns the array `error` of shape `(N,)`, where `error[k]` is the absolute error of `preds[k]` of order `ord` from `targets[k]` for `k` in `range(N)`.
    
    -----
    ### Note
    To compute the error using the maximum function, pass `ord='inf'`.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if dim is None:
        dim = tuple(range(1, preds.ndim))
    if isinstance(p, str):
        p = p.lower()
        if p == 'inf':
            p = torch.inf
    if scale is None:
        scale: torch.Tensor = torch.ones((preds.size(0),), device=preds.device)
    diff        = torch.norm(preds-targets, dim=dim, p=p)
    scale       = scale.reshape(diff.shape)
    abs_error   = scale * diff
    return abs_error


def relative_error(
        preds:      torch.Tensor,
        targets:    torch.Tensor,
        p:          Union[float, str] = 2.0,
        dim:        Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Returns the instance-wise relative error between the `preds` and `targets`.
    
    -----
    Given to sequences `preds` and `targets` of shape `(N, ...)` (where `N` is the number of the instances), this function returns the array `error` of shape `(N,)`, where `error[k]` is the relative error of `preds[k]` of order `ord` from `targets[k]` for `k` in `range(N)`.
    
    -----
    ### Note
    To compute the error using the maximum function, pass `ord='inf'`.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if dim is None:
        dim = tuple(range(1, preds.ndim))
    if isinstance(p, str):
        p = p.lower()
        if p == 'inf':
            p = torch.inf
    numer = torch.norm(preds-targets, dim=dim, p=p)
    denom = torch.norm(targets, dim=dim, p=p)
    return numer/denom


##################################################
##################################################
# End of file