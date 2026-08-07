import  torch
from    scipy.special   import  gamma


__all__: list[str] = ['sinc', 'phase', 'area_of_unit_sphere', 'volume_of_unit_ball']


##################################################
##################################################
def sinc(x: torch.Tensor) -> torch.Tensor:
    """Returns `sin(x)/x` for `x!=0` and `1` for `x==0`."""
    return torch.where(
        x!=0,
        torch.sin(x)/x,
        torch.ones_like(x, dtype=x.dtype, device=x.device),
    )


def phase(theta: torch.Tensor) -> torch.Tensor:
    """Returns `exp(1j * theta)`."""
    return torch.exp(1j * theta)


def area_of_unit_sphere(dim_embed: int) -> float:
    r"""Returns the area of the unit sphere $S^{d-1}$ embeded in $\mathbb{R}^d$."""
    return 2 * (torch.pi**(dim_embed/2)) / gamma(dim_embed/2)


def volume_of_unit_ball(dim_embed: int) -> float:
    r"""Returns the volume of the unit ball embeded in $\mathbb{R}^d$."""
    return torch.pi**(dim_embed/2) / gamma(1 + dim_embed/2)


##################################################
##################################################
# End of file