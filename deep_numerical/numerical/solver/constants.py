__all__: list[str] = [
    'LAMBDA',
    'LAMBDA_CARLEMAN',
    'LAMBDA_FPL',
    'DEFAULT_QUAD_ORDER_UNIFORM',
    'DEFAULT_QUAD_ORDER_LEGENDRE',
    'DEFAULT_QUAD_ORDER_LEBEDEV',
]

# Constants for the spectral method
LAMBDA:          float = 2 / (3 + (2 ** 0.5))
r"""The least required ratio `2/(3+sqrt(2)) \approx 0.4531` of period to support diameter in the Fourier-Galerkin method for the Boltzmann equation."""

LAMBDA_CARLEMAN: float = 2 / (1 + (18 ** 0.5))
r"""The least required ratio `2/(1+sqrt(18)) \approx 0.3815` in the Carleman-representation spectral method."""

LAMBDA_FPL:      float = 0.5
"""The least required ratio `0.5` of period to support diameter for the Fokker-Planck-Landau equation."""

# Constants for numerical integration
DEFAULT_QUAD_ORDER_UNIFORM:  int = 30
DEFAULT_QUAD_ORDER_LEGENDRE: int = 20
DEFAULT_QUAD_ORDER_LEBEDEV:  int = 7


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()