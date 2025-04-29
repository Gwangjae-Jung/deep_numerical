from    math        import  sqrt


##################################################
##################################################
__all__: list[str] = ['EPSILON', 'LAMBDA', 'LAMBDA_CARLEMAN', 'LAMBDA_FPL']


##################################################
##################################################
# Constants for preventing computational errors
EPSILON:   float   = 1e-20


# Constants for the spectral method
LAMBDA:             float = 2 / (3 + sqrt(2))
"""
This is the value `2 / (3+sqrt(2))`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Boltzmann equation.
"""
LAMBDA_CARLEMAN:    float = 2 / (1 + sqrt(18))
"""
This is the value `2 / (1+3*sqrt(2))`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Boltzmann equation, in which the Carleman-like representation is involved.
"""
LAMBDA_FPL:         float = 0.5
"""
This is the value `0.5`, which is the least required ratio of the period and the diameter of the support of the distribution function in the Fourier-Galerkin method (the spectral method) for solving the Fokker-Planck-Landau equation.
"""


##################################################
##################################################
# End of file