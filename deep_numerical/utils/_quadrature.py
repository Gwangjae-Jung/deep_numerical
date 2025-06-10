import  torch
from    scipy.special       import  roots_legendre
from    scipy.integrate     import  lebedev_rule
from    ._dtype             import  TORCH_DEFAULT_DTYPE, TORCH_DEFAULT_DEVICE


##################################################
##################################################
__all__: list[str] = [
    'DEFAULT_QUAD_ORDER_UNIFORM',
    'DEFAULT_QUAD_ORDER_LEGENDRE',
    'DEFAULT_QUAD_ORDER_LEBEDEV',
    'roots_uniform_shifted',
    'roots_linspace',
    'roots_legendre_shifted',
    'roots_lebedev',
    'roots_circle',
    'polar_grid',
    'spherical_grid',
    'roots_uniform',
]


##################################################
##################################################
# Quadrature and grid
DEFAULT_QUAD_ORDER_UNIFORM:     int = 30
DEFAULT_QUAD_ORDER_LEGENDRE:    int = 20
DEFAULT_QUAD_ORDER_LEBEDEV:     int = 7
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


def _check_interval(n: int, a: float, b: float) -> None:
    if n <= 1:
        raise ValueError(
            f"'n' should be a positive integer greater than 1.\n"
            f"'n': {n}"
        )
    if a >= b:
        raise ValueError(
            f"'a' should be smaller than 'b'.\n"
            f"'a': {a:.4e}\n'b': {b:.4e}"
        )


def roots_uniform_shifted(
        n:              int,
        a:              float,
        b:              float,
        is_symmetric:   bool = False,
        dtype:          torch.dtype     = TORCH_DEFAULT_DTYPE,
        device:         torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of the Legendre quadrature points and weights."""
    _check_interval(n, a, b)
    delta = (b-a) / n
    roots: torch.Tensor
    if is_symmetric:
        roots = torch.linspace(a+delta/2, b-delta/2, n, dtype=dtype)
    else:
        roots = a + delta*torch.arange(n, dtype=dtype, device=device)
    weights = delta * torch.ones_like(roots, dtype=dtype, device=device)
    return (roots, weights)


roots_uniform = roots_uniform_shifted


def roots_linspace(
        n:      int,
        a:      float,
        b:      float,
        dtype:  torch.dtype     = TORCH_DEFAULT_DTYPE,
        device: torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of the uniform quadrature points and weights."""
    _check_interval(n, a, b)
    roots = torch.linspace(a, b, n, dtype=dtype, device=device)
    weights = (b-a) * torch.ones_like(roots, dtype=dtype, device=device) / (n-1)
    return (roots, weights)


def roots_legendre_shifted(
        n:      int,
        a:      float,
        b:      float,
        dtype:  torch.dtype     = TORCH_DEFAULT_DTYPE,
        device: torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns 1-dimensional arrays of the Legendre quadrature points and weights."""
    _check_interval(n, a, b)
    roots, weights = roots_legendre(n)
    roots   = torch.tensor((a+b)/2 + (b-a)*roots/2, dtype=dtype, device=device)
    weights = torch.tensor((b-a)*weights/2, dtype=dtype, device=device)
    return (roots, weights)


def roots_lebedev(
        order:  int,
        dtype:  torch.dtype     = TORCH_DEFAULT_DTYPE,
        device: torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a 2-dimensional array of the Lebedev quadrature points on the unit sphere of shape `(N, 3)` and a 1-dimensional array of weights, where `N` is the number of points in the quadrature."""
    roots, weights = lebedev_rule(order)
    roots   = torch.tensor(roots,   dtype=dtype, device=device).transpose(1, 0)
    weights = torch.tensor(weights, dtype=dtype, device=device)
    return roots, weights


def roots_circle(
        n:      int,
        dtype:  torch.dtype     = TORCH_DEFAULT_DTYPE,
        device: torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a 2-dimensional array of the uniform quadrature points on the unit circle of shape `(N, 2)` and a 1-dimensional array of weights, where `N` is the number of points in the quadrature."""
    _thetas, weights = roots_uniform(n, 0, 2*torch.pi, dtype=dtype, device=device)
    roots = torch.stack((torch.cos(_thetas), torch.sin(_thetas)), dim=-1)
    return (roots, weights)


def polar_grid(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    if radius.ndim != 1 or angle.ndim != 1:
        raise RuntimeError(
            f"Check the dimensions of the input arrays:\n"
            f"* radius.ndim: {radius.ndim}\n"
            f"* angle.ndim:  {angle.ndim}"
        )
    r = radius.reshape(-1, 1)
    t = angle.reshape( 1, -1)
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    grid = torch.stack((x, y), dim=-1)
    return grid
        

def spherical_grid(
        radius:             torch.Tensor,
        polar_angle:        torch.Tensor,
        azimuthal_angle:    torch.Tensor,
    ) -> torch.Tensor:
    if radius.ndim != 1 or polar_angle.ndim != 1 or azimuthal_angle.ndim != 1:
        raise RuntimeError(
            f"Check the dimensions of the input arrays:\n"
            f"* radius.ndim:          {radius.ndim}\n"
            f"* polar_angle.ndim:     {polar_angle.ndim}\n"
            f"* azimuthal_angle.ndim: {azimuthal_angle.ndim}"
        )
    rho     = radius.reshape(         -1, 1, 1)
    phi     = polar_angle.reshape(    1, -1, 1)
    theta   = azimuthal_angle.reshape(1, 1, -1)
    _xy = rho * torch.sin(phi)
    x   = _xy * torch.cos(theta)
    y   = _xy * torch.sin(theta)
    z   = rho * torch.cos(phi)
    z   = torch.tile(z, reps=(1, 1, len(azimuthal_angle)))
    grid = torch.stack((x, y, z), dim=-1)
    return grid


##################################################
##################################################
# End of file