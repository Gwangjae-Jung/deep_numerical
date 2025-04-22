from    typing          import  Optional, Sequence, Union
import  numpy           as      np
import  torch
from    scipy.special   import  gamma


##################################################
##################################################
__all__: list[str] = [
    'sinc',
    'phase',
    'area_of_unit_sphere',
    'volume_of_unit_ball',
    'metric',
]


##################################################
##################################################
def sinc(x: np.ndarray) -> np.ndarray:
    """Returns `sin(x)/x`."""
    with np.errstate(invalid='ignore'):
        return np.where(x!=0, np.sin(x)/x, np.ones_like(x))


def phase(theta: np.ndarray) -> np.ndarray:
    """Returns `exp(i * theta)`."""
    return np.exp(1j * theta)


##################################################
##################################################
def area_of_unit_sphere(dim_embed: int) -> float:
    r"""Returns the area of the unit sphere $S^{d-1}$ embeded in $\mathbb{R}^d$."""
    return 2 * (np.pi**(dim_embed/2)) / gamma(dim_embed/2)


def volume_of_unit_ball(dim_embed: int) -> float:
    r"""Returns the volume of the unit ball embeded in $\mathbb{R}^d$."""
    return np.pi**(dim_embed/2) / gamma(1 + dim_embed/2)


##################################################
##################################################
def metric(
        preds:      Union[np.ndarray, torch.Tensor],
        targets:    Union[np.ndarray, torch.Tensor],
        axes:       Optional[Sequence[int]] = None,
        ord:        Union[float, str] = 2.0
    ) -> Union[np.ndarray, torch.Tensor]:
    """Returns the instance-wise metric between the `preds` and `targets`.
    
    -----
    Given to sequences `preds` and `targets` of shape `(N, ...)` (where `N` is the number of the instances), this function returns the array `error` of shape `(N,)`, where `error[k]` is the relative error of `preds[k]` of order `ord` from `targets[k]` for `k` in `range(N)`.
    
    -----
    ### Note
    To compute the error using the maximum function, pass `ord='inf'`.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if axes is None:
        axes = tuple(range(1, preds.ndim))
    if isinstance(ord, str):
        ord = ord.lower()
        if ord == 'inf':
            ord = torch.inf
    preds   = torch.from_numpy(preds).type(torch.float) if isinstance(preds, np.ndarray) else preds
    targets = torch.from_numpy(targets).type(torch.float) if isinstance(targets, np.ndarray) else targets
    numer = torch.norm(preds-targets, dim=axes, p=ord)
    denom = torch.norm(targets, dim=axes, p=ord)
    return (numer/denom).numpy()


##################################################
##################################################
# End of file