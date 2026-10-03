from   typing          import Optional
import torch
from   scipy.integrate import lebedev_rule
from   scipy.special   import roots_legendre


__all__: list[str] = [
    'roots_uniform_shifted',
    'roots_linspace',
    'roots_legendre_shifted',
    'roots_lebedev',
    'roots_circle',
    'polar_grid',
    'spherical_grid',
]


DEFAULT_QUAD_ORDER_UNIFORM:  int = 30
DEFAULT_QUAD_ORDER_LEGENDRE: int = 20
DEFAULT_QUAD_ORDER_LEBEDEV:  int = 7
"""
### Note
The following is the collection of the pairs of the degree of the Lebedev quadrature and the number of points in the quadrature, supported by the function `scipy.integrate.lebedev_rule`.\n
`(3, 6)`\n
`(5, 14)`\n
`(7, 26)`\n
`(9, 38)`\n
`(11, 50)`\n
`(13, 74)`\n
`(15, 86)`\n
`(17, 110)`\n
`(19, 146)`\n
`(21, 170)`\n
`(23, 194)`\n
`(25, 230)`\n
`(27, 266)`\n
`(29, 302)`\n
`(31, 350)`\n
`(35, 434)`\n
`(41, 590)`\n
`(47, 770)`\n
`(53, 974)`\n
`(59, 1202)`\n
`(65, 1454)`\n
`(71, 1730)`\n
`(77, 2030)`\n
`(83, 2354)`\n
`(89, 2702)`\n
`(95, 3074)`\n
`(101, 3470)`\n
`(107, 3890)`\n
`(113, 4334)`\n
`(119, 4802)`\n
`(125, 5294)`\n
`(131, 5810)`\n
"""


##################################################
def _check_interval(n: int, a: float, b: float) -> None:
    """Validates the input interval and number of quadrature points.

    ## Description
    Ensures that the number of quadrature points `n` is greater than 1, and that the lower bound `a` is strictly less than the upper bound `b`.

    ## Arguments
    `n` (`int`): Number of quadrature points. Must be greater than 1.
    `a` (`float`): Lower bound of the interval.
    `b` (`float`): Upper bound of the interval.

    ## Returns
    `None`: Returns `None` if validation passes, otherwise raises `ValueError`.
    """
    if n <= 1:
        raise ValueError(f"'n' should be a positive integer greater than 1, but [{n=}].")
    if a >= b:
        raise ValueError(f"'a' should be smaller than 'b', but [{a=}, {b=}].")
    return None


def roots_uniform_shifted(
        n:            int,
        a:            float,
        b:            float,
        is_symmetric: bool                   = False,
        dtype:        Optional[torch.dtype]  = None,
        device:       Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of uniform quadrature points and weights.

    ## Description
    Computes shifted uniform quadrature points and uniform weights on the interval `[a, b]`.
    If `is_symmetric` is `True`, points are centered within sub-intervals (midpoint rule);
    otherwise, points start at `a`.

    ## Arguments
    `n` (`int`): The number of quadrature points.
    `a` (`float`): Lower bound of the interval.
    `b` (`float`): Upper bound of the interval.
    `is_symmetric` (`bool`, default: `False`): Whether to use centered midpoint points.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensors.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensors.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: A tuple containing `(roots, weights)`.
    """
    _check_interval(n, a, b)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    delta: float = (b - a) / n
    roots: torch.Tensor
    if is_symmetric:
        roots = torch.linspace(a + delta / 2, b - delta / 2, n, dtype=dtype, device=device)
    else:
        roots = a + delta * torch.arange(n, dtype=dtype, device=device)
    weights: torch.Tensor = delta * torch.ones_like(roots, dtype=dtype, device=device)
    return (roots, weights)


def roots_linspace(
        n:      int,
        a:      float,
        b:      float,
        dtype:  Optional[torch.dtype]  = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of evenly spaced quadrature points and trapezoidal-like uniform weights.

    ## Description
    Generates `n` linearly spaced points on `[a, b]` with uniform weights `(b - a) / (n - 1)`.

    ## Arguments
    `n` (`int`): The number of points.
    `a` (`float`): Lower bound of the interval.
    `b` (`float`): Upper bound of the interval.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensors.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensors.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: A tuple containing `(roots, weights)`.
    """
    _check_interval(n, a, b)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    roots:   torch.Tensor = torch.linspace(a, b, n, dtype=dtype, device=device)
    weights: torch.Tensor = (b - a) * torch.ones_like(roots, dtype=dtype, device=device) / (n - 1)
    return (roots, weights)


def roots_legendre_shifted(
        n:      int,
        a:      float,
        b:      float,
        dtype:  Optional[torch.dtype]  = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of Gauss-Legendre quadrature points and weights shifted to `[a, b]`.

    ## Description
    Calculates the roots and weights for Gauss-Legendre quadrature of order `n`, affine-transformed from `[-1, 1]` to the interval `[a, b]`.

    ## Arguments
    `n` (`int`): The quadrature order.
    `a` (`float`): Lower bound of the interval.
    `b` (`float`): Upper bound of the interval.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensors.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensors.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: A tuple containing `(roots, weights)`.
    """
    _check_interval(n, a, b)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    raw_roots, raw_weights = roots_legendre(n)
    roots:   torch.Tensor = torch.tensor((a + b) / 2 + (b - a) * raw_roots / 2, dtype=dtype, device=device)
    weights: torch.Tensor = torch.tensor((b - a) * raw_weights / 2, dtype=dtype, device=device)
    return (roots, weights)


def roots_lebedev(
        order:  int,
        dtype:  Optional[torch.dtype]  = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns Lebedev quadrature points on the unit sphere and their corresponding weights.

    ## Description
    Computes Lebedev quadrature points on `S^2` of shape `(N, 3)` and weights of shape `(N,)` for a given order, where `N` is the number of points.

    ## Arguments
    `order` (`int`): The degree / order of the Lebedev quadrature rule.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensors.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensors.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: A tuple of `(roots, weights)` on `S^2`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    raw_roots, raw_weights = lebedev_rule(order)
    roots:   torch.Tensor = torch.tensor(raw_roots,   dtype=dtype, device=device).transpose(1, 0)
    weights: torch.Tensor = torch.tensor(raw_weights, dtype=dtype, device=device)
    return (roots, weights)


def roots_circle(
        n:      int,
        dtype:  Optional[torch.dtype]  = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns uniform quadrature points on the unit circle `S^1` and their weights.

    ## Description
    Generates `n` uniformly distributed points on the unit circle in 2D Cartesian coordinates `(x, y)` with uniform integration weights.

    ## Arguments
    `n` (`int`): The number of quadrature points on the circle.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensors.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensors.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: A tuple `(roots, weights)` where `roots` has shape `(n, 2)` and `weights` has shape `(n,)`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    _thetas, weights = roots_uniform_shifted(n, 0.0, 2.0 * torch.pi, dtype=dtype, device=device)
    roots: torch.Tensor = torch.stack((torch.cos(_thetas), torch.sin(_thetas)), dim=-1)
    return (roots, weights)


def polar_grid(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Constructs a 2D Cartesian coordinate grid from radial and angular coordinate tensors.

    ## Description
    Constructs a 2D Cartesian coordinate grid from 1D radial and angular coordinate tensors.

    ## Arguments
    `radius` (`torch.Tensor`): A 1D tensor containing radial coordinates.
    `angle` (`torch.Tensor`): A 1D tensor containing polar angles in radians.

    ## Returns
    `torch.Tensor`: A tensor of shape `(len(radius), len(angle), 2)` containing `(x, y)` Cartesian coordinates.
    """
    if radius.ndim != 1 or angle.ndim != 1:
        raise RuntimeError(
            f"Check the dimensions of the input arrays:\n"
            f"* radius.ndim: {radius.ndim}\n"
            f"* angle.ndim:  {angle.ndim}"
        )
    r:    torch.Tensor = radius.reshape(-1, 1)
    t:    torch.Tensor = angle.reshape(1, -1)
    x:    torch.Tensor = r * torch.cos(t)
    y:    torch.Tensor = r * torch.sin(t)
    grid: torch.Tensor = torch.stack((x, y), dim=-1)
    return grid


def spherical_grid(
        radius:          torch.Tensor,
        polar_angle:     torch.Tensor,
        azimuthal_angle: torch.Tensor,
    ) -> torch.Tensor:
    """Constructs a 3D Cartesian coordinate grid from radial, polar, and azimuthal angle tensors.

    ## Description
    Constructs a 3D Cartesian coordinate grid from 1D radial, polar (zenith), and azimuthal angle tensors.

    ## Arguments
    `radius` (`torch.Tensor`): A 1D tensor containing radial coordinates.
    `polar_angle` (`torch.Tensor`): A 1D tensor containing polar (zenith) angles `phi` in radians.
    `azimuthal_angle` (`torch.Tensor`): A 1D tensor containing azimuthal angles `theta` in radians.

    ## Returns
    `torch.Tensor`: A tensor of shape `(len(radius), len(polar_angle), len(azimuthal_angle), 3)` containing `(x, y, z)` Cartesian coordinates.
    """
    if radius.ndim != 1 or polar_angle.ndim != 1 or azimuthal_angle.ndim != 1:
        raise RuntimeError(
            f"Check the dimensions of the input arrays:\n"
            f"* radius.ndim:          {radius.ndim}\n"
            f"* polar_angle.ndim:     {polar_angle.ndim}\n"
            f"* azimuthal_angle.ndim: {azimuthal_angle.ndim}"
        )
    rho:   torch.Tensor = radius.reshape(-1, 1, 1)
    phi:   torch.Tensor = polar_angle.reshape(1, -1, 1)
    theta: torch.Tensor = azimuthal_angle.reshape(1, 1, -1)
    _xy:   torch.Tensor = rho * torch.sin(phi)
    x:     torch.Tensor = _xy * torch.cos(theta)
    y:     torch.Tensor = _xy * torch.sin(theta)
    z:     torch.Tensor = rho * torch.cos(phi)
    z = torch.tile(z, reps=(1, 1, len(azimuthal_angle)))
    grid:  torch.Tensor = torch.stack((x, y, z), dim=-1)
    return grid


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()