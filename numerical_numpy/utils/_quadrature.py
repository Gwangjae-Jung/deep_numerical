import  numpy               as      np
from    scipy.special       import  roots_legendre


##################################################
##################################################
__all__: list[str] = [
    'DEFAULT_QUAD_ORDER_UNIFORM',
    'DEFAULT_QUAD_ORDER_LEGENDRE',
    'DEFAULT_QUAD_ORDER_LEBEDEV',
    'roots_uniform_shifted',
    'roots_legendre_shifted',
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
    ) -> tuple[np.ndarray, np.ndarray]:
    _check_interval(n, a, b)
    delta = (b-a) / n
    roots: np.ndarray
    if is_symmetric:
        roots = np.linspace(a+delta/2, b-delta/2, n)
    else:
        roots = a + delta*np.arange(n)
    weights = delta * np.ones_like(roots)
    return (roots, weights)

def roots_linspace(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    _check_interval(n, a, b)
    roots = np.linspace(a, b, n)
    weights = (b-a) * np.ones_like(roots) / (n-1)
    return (roots, weights)


def roots_legendre_shifted(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    _check_interval(n, a, b)
    roots, weights = roots_legendre(n)
    roots   = (a+b)/2 + (b-a)*roots/2
    weights = (b-a)*weights/2
    return (roots, weights)


def roots_circle(n: int) -> tuple[np.ndarray, np.ndarray]:
    _thetas, weights = roots_uniform(n, 0, 2*np.pi)
    roots = np.stack((np.cos(_thetas), np.sin(_thetas)), axis=-1)
    return (roots, weights)


def polar_grid(radius: np.ndarray, angle:  np.ndarray) -> np.ndarray:
    if radius.ndim != 1 or angle.ndim != 1:
        raise RuntimeError(
            f"Check the dimensions of the input arrays:\n"
            f"* radius.ndim: {radius.ndim}\n"
            f"* angle.ndim:  {angle.ndim}"
        )
    r = radius.reshape(-1, 1)
    t = angle.reshape( 1, -1)
    x = r * np.cos(t)
    y = r * np.sin(t)
    grid = np.stack((x, y), axis=-1)
    return grid
        

def spherical_grid(
        radius:             np.ndarray,
        polar_angle:        np.ndarray,
        azimuthal_angle:    np.ndarray,
    ) -> np.ndarray:
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
    _xy = rho * np.sin(phi)
    x   = _xy * np.cos(theta)
    y   = _xy * np.sin(theta)
    z   = rho * np.cos(phi)
    z   = np.tile(z, reps=(1, 1, len(azimuthal_angle)))
    grid = np.stack((x, y, z), axis=-1)
    return grid


roots_uniform = roots_uniform_shifted


##################################################
##################################################
# End of file