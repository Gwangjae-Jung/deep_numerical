from   scipy.special import gamma
import torch


__all__: list[str] = ['sinc', 'phase', 'area_of_unit_sphere', 'volume_of_unit_ball']


##################################################
def sinc(x: torch.Tensor) -> torch.Tensor:
    """Returns normalized sinc function sin(x)/x.

    ## Description
    Returns `sin(x) / x` for `x != 0` and `1` for `x == 0`.

    ## Arguments
    `x` (`torch.Tensor`): Input tensor.

    ## Returns
    `torch.Tensor`: Elementwise evaluated sinc tensor.
    """
    return torch.where(
        x != 0,
        torch.sin(x) / x,
        torch.ones_like(x, dtype=x.dtype, device=x.device),
    )


def phase(theta: torch.Tensor) -> torch.Tensor:
    """Returns complex phase exp(1j * theta).

    ## Description
    Computes the complex phase exponential `exp(1j * theta)`.

    ## Arguments
    `theta` (`torch.Tensor`): Phase angle tensor.

    ## Returns
    `torch.Tensor`: Complex exponential tensor.
    """
    return torch.exp(1j * theta)


def area_of_unit_sphere(dim_embed: int) -> float:
    r"""Returns the surface area of the unit sphere.

    ## Description
    Returns the area of the unit sphere $S^{d-1}$ embedded in $\mathbb{R}^d$.

    ## Arguments
    `dim_embed` (`int`): Dimension `d` of the embedding space.

    ## Returns
    `float`: Surface area of $S^{d-1}$.
    """
    return 2.0 * float(torch.pi ** (dim_embed / 2)) / float(gamma(dim_embed / 2))


def volume_of_unit_ball(dim_embed: int) -> float:
    r"""Returns the volume of the unit ball.

    ## Description
    Returns the volume of the unit ball embedded in $\mathbb{R}^d$.

    ## Arguments
    `dim_embed` (`int`): Dimension `d` of the embedding space.

    ## Returns
    `float`: Volume of the unit ball.
    """
    return float(torch.pi ** (dim_embed / 2)) / float(gamma(1 + dim_embed / 2))


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()