import  numpy               as      np
import  torch
from    scipy.special       import  roots_legendre


##################################################
##################################################
__all__: list[str] = [
    'DEFAULT_QUAD_ORDER_UNIFORM',
    'DEFAULT_QUAD_ORDER_LEGENDRE',
    'DEFAULT_QUAD_ORDER_LEBEDEV',
    'roots_uniform_shifted',
    'roots_linspace',
    'roots_legendre_shifted',
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
    _check_interval(n, a, b)
    delta = (b-a) / n
    roots: np.ndarray
    if is_symmetric:
        roots = np.linspace(a+delta/2, b-delta/2, n)
    else:
        roots = a + delta*np.arange(n)
    weights = delta * np.ones_like(roots)
    return tuple(map(lambda x: torch.tensor(x, dtype=torch.float), (roots, weights)))


def roots_linspace(n: int, a: float, b: float) -> tuple[torch.Tensor, torch.Tensor]:
    # _check_interval(n, a, b)
    roots = np.linspace(a, b, n)
    weights = (b-a) * np.ones_like(roots) / (n-1) if n>1 else np.array([1.0])
    return tuple(map(lambda x: torch.tensor(x, dtype=torch.float), (roots, weights)))


def roots_legendre_shifted(n: int, a: float, b: float) -> tuple[torch.Tensor, torch.Tensor]:
    _check_interval(n, a, b)
    roots, weights = roots_legendre(n)
    roots   = (a+b)/2 + (b-a)*roots/2
    weights = (b-a)*weights/2
    return tuple(map(lambda x: torch.tensor(x, dtype=torch.float), (roots, weights)))


roots_uniform = roots_uniform_shifted


##################################################
##################################################
# End of file