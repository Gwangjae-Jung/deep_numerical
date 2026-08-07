__all__: list[str] = [
    'LAMBDA', 'LAMBDA_CARLEMAN', 'LAMBDA_FPL',
    'DEFAULT_QUAD_ORDER_UNIFORM', 'DEFAULT_QUAD_ORDER_LEGENDRE', 'DEFAULT_QUAD_ORDER_LEBEDEV',
]


# Constants for the spectral method
LAMBDA:             float = 2/(3+(2**0.5))
r"""This is the value `2/(3+sqrt(2)) \approx 0.4531`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Boltzmann equation.
"""
LAMBDA_CARLEMAN:    float = 2/(1+(18**0.5))
r"""This is the value `2/(1+sqrt(18)) \approx 0.3815`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Boltzmann equation, in which the Carleman-like representation is involved.
"""
LAMBDA_FPL:         float = 0.5
"""This is the value `0.5`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Fokker-Planck-Landau equation.
"""


# Constants for numerical integration
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


##################################################
##################################################
# End of file