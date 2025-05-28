"""
### Note
All numerical integrators in this script assume that the numeric input `r` of `func` (the input except for those in `func_kwargs`) is a 1-dimensional array.
To conduct numerical integration, a reshape of `r` should be implemented inside `func`.
"""
import  warnings
import  numpy           as      np
from    typing          import  Callable, Optional
from    scipy.integrate import  lebedev_rule

from    ._quadrature    import  DEFAULT_QUAD_ORDER_LEBEDEV, roots_legendre_shifted


##################################################
##################################################
__all__: list[str] =  [
    'integration_guass_legendre',
    'integration_lebedev',
    'integration_legendre',
    'integration_S2',
    
    'integration_closed_Newton_Cotes',
    'integration_open_Newton_Cotes',
    'integration_closed_Newton_Cotes_composite',
    'integration_romberg_backward',
    'integration_romberg',
]


##################################################
##################################################
# Numerical integration
def integration_guass_legendre(
        num_roots:      int,
        a:              float,
        b:              float,
        func:           Callable[[np.ndarray, object], np.ndarray],
        func_kwargs:    Optional[dict[str, object]] = {},   
    ) -> np.ndarray:
    """Numerical integration on a compact interval using the Gauss-Legendre quadrature rule
    
    -----
    ### Description
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the Gauss-Legendre quadrature rule of order `num_roots`.
    
    -----
    ### Arguments
        * `num_roots` (`int`)
            The number of roots of the Gauss-Legendre quadrature rule.
        * `a` (`float`)
            The lower bound of the interval.
        * `b` (`float`)
            The upper bound of the interval.
        * `func` (`Callable[[np.ndarray, object], np.ndarray]`)
            The integrand. As mentioned above, the numeric input `r` of `func` should be a 1-dimensional array, and its reshape should be implemented inside `func`.
        * `func_kwargs` (`Optional[dict[str, object]]`)
            The other arguments of `func`.
    """
    roots, weights = roots_legendre_shifted(num_roots, a, b)
    func_vals = func(roots, **func_kwargs)
    return np.einsum("...t,t->...", func_vals, weights)


def integration_lebedev(
        f:                  Callable[[np.ndarray], float],
        quad_order_lebedev: int = DEFAULT_QUAD_ORDER_LEBEDEV,
    ) -> np.ndarray:
    """Numerical integration on S^2 using the Lebedev quadrature rule
    -----
    
    ### Description
    This function supports the numerical integration of a complex-valued function `f`, using the Lebedev quadrature rule of order `quad_order_lebedev`.
    
    -----
    ### Note
    Currently, the parallel computation is not implemented, which is easy to be implemented.
    """
    weights: np.ndarray
    roots, weights = lebedev_rule(quad_order_lebedev)
    roots = roots.transpose(1, 0)
    return np.sum(f(roots) * weights)


integration_legendre    = integration_guass_legendre
integration_S2          = integration_lebedev


##################################################
##################################################
def integration_closed_Newton_Cotes(
        a:              float,
        b:              float,
        func:           Callable[[np.ndarray], np.ndarray],
        func_kwargs:    Optional[dict[str, object]] = {},
        degree:         int = 4,
    ) -> np.ndarray:
    """Numerical integration using the closed Newton-Cotes formula
    
    -----
    ### Description
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the closed Newton-Cotes formula of degree `degree`.
    Here, the number of the grid points on which `func` is evaluated is `degree+1`.
    The closed Newton-Cotes formula uses a uniform discretization of the compact interval `[a, b]` and Lagrange interpolation polynomials, and is exact for
        * polynomials of degree at most `degree` if `degree` is odd,
        * polynomials of degree at most `degree+1` if `degree` is even.
    
    This function provides four types of closed Newton-Cotes formulas.
    In each of the following formulas, the step size `h` is defined as `(b-a)/degree`.
    1. Trapezoid rule (`degree = 1`)
        * Exact for polynomials of degree not greater than 1.
        * Error term of order `h**3`.
    2. Simpson's rule (`degree = 2`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `h**5`.
    3. Simpson's 3/8 rule (`degree = 3`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `h**5`.
    4. Boole's rule (`degree = 4`)
        * Exact for polynomials of degree not greater than 5.
        * Error term of order `h**7`.
    
    Reference: [Wikipedia](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas)
    
    -----
    ### Implementation
    The closed Newton-Cotes formulas have the following form:
        `coeff_h * h * (linear_combination_of_func_values)`
    This function implements the above equation, where `coeff_h` and the coefficients of the linear combination are computed in other functions.
    """
    _supported_degrees = set((1, 2, 3, 4))
    if degree not in _supported_degrees:
        raise ValueError(f"Unsupported degree: {degree}. Supported degrees: {_supported_degrees}.")
    if a > b:
        a, b = b, a
    grid_points = np.linspace(a, b, degree+1)
    step_size = float((b-a) / degree)
    
    coeff_h, coeff_LC = _closed_Newton_Cotes_coefficients(degree)
    linear_comb = np.einsum("i,...i->...", coeff_LC, func(grid_points, **func_kwargs))
    return coeff_h * step_size * linear_comb


def _closed_Newton_Cotes_coefficients(degree: int) -> tuple[float, np.ndarray]:
    if degree == 1:
        return (0.5, np.array([1, 1], dtype=np.float64))
    elif degree == 2:
        return (1/3, np.array([1, 4, 1], dtype=np.float64))
    elif degree == 3:
        return (3/8, np.array([1, 3, 3, 1], dtype=np.float64))
    elif degree == 4:
        return (2/45, np.array([7, 32, 12, 32, 7], dtype=np.float64))
    else:
        raise NotImplementedError(f"Unsupported degree: {degree}.")


def integration_open_Newton_Cotes(
        a:              float,
        b:              float,
        func:           Callable[[np.ndarray], np.ndarray],
        func_kwargs:    Optional[dict[str, object]] = {},
        degree:         int = 3,
    ) -> np.ndarray:
    """Numerical integration using the open Newton-Cotes formula
    
    -----
    ### Description
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the open Newton-Cotes formula of degree `degree`.
    Here, the number of the grid points on which `func` is evaluated is `degree+1`.
    The open Newton-Cotes formula uses a uniform discretization of the compact interval `[a, b]` without the endpoints `a` and `b`, and Lagrange interpolation polynomials, and is exact for
        * polynomials of degree at most `degree` if `degree` is odd,
        * polynomials of degree at most `degree+1` if `degree` is even.
    
    This function provides four types of open Newton-Cotes formulas.
    In each of the following formulas, the step size `h` is defined as `(b-a)/(degree+2)` - note that both endpoints are ignored.
    1. Midpoint rule (`degree = 0`)
        * Exact for polynomials of degree not greater than 1.
        * Error term of order `h**3`.
    2. Name unknown (`degree = 1`)
        * Exact for polynomials of degree not greater than 1.
        * Error term of order `h**3`.
    3. Milne's rule (`degree = 2`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `h**5`.
    4. Name unknown (`degree = 3`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `h**5`.
    
    Reference: [Wikipedia](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Open_Newton%E2%80%93Cotes_formulas)
    
    -----
    ### Implementation
    The open Newton-Cotes formulas have the following form:
        `coeff_h * h * (linear_combination_of_func_values)`
    This function implements the above equation, where `coeff_h` and the coefficients of the linear combination are computed in other functions.
    """
    _supported_degrees = set((0, 1, 2, 3))
    if degree not in _supported_degrees:
        raise ValueError(f"Unsupported degree: {degree}. Supported degrees: {_supported_degrees}.")
    if a > b:
        a, b = b, a
    grid_points = np.linspace(a, b, degree+3)[1:-1]
    step_size = float((b-a) / (degree+2))
    
    coeff_h, coeff_LC = _open_Newton_Cotes_coefficients(degree)
    linear_comb = np.einsum("i,...i->...", coeff_LC, func(grid_points, **func_kwargs))
    return coeff_h * step_size * linear_comb


def _open_Newton_Cotes_coefficients(degree: int) -> tuple[float, np.ndarray]:
    if degree == 0:
        return (2, np.array([1], dtype=np.float64))
    elif degree == 1:
        return (3/2, np.array([1, 1], dtype=np.float64))
    elif degree == 2:
        return (4/3, np.array([2, -1, 2], dtype=np.float64))
    elif degree == 3:
        return (5/24, np.array([11, 1, 1, 11], dtype=np.float64))
    else:
        raise ValueError(f"Unsupported degree: {degree}.")



def integration_closed_Newton_Cotes_composite(
        a:                  float,
        b:                  float,
        func:               Callable[[np.ndarray], np.ndarray],
        func_kwargs:        Optional[dict[str, object]] = {},
        degree:             int = 4,
        num_subintervals:   int = 2,
    ) -> np.ndarray:
    """Numerical integration using the closed Newton-Cotes formula (composite version)
    
    -----
    ### Description
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the closed Newton-Cotes formula of degree `degree`.
    Here, the number of the grid points on which `func` is evaluated is `degree+1`.
    The closed Newton-Cotes formula uses a uniform discretization of the compact interval `[a, b]` and Lagrange interpolation polynomials, and is exact for
        * polynomials of degree at most `degree` if `degree` is odd,
        * polynomials of degree at most `degree+1` if `degree` is even.
    
    This function provides four types of closed Newton-Cotes formulas.
    In each of the following formulas, the step size `h` is defined as `(b-a)/degree`, and we denote `num_subintervals` by `n`.
    1. Trapezoid rule (`degree = 1`)
        * Exact for polynomials of degree not greater than 1.
        * Error term of order `1/n**2`.
    2. Simpson's rule (`degree = 2`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `1/n**4`.
    3. Simpson's 3/8 rule (`degree = 3`)
        * Exact for polynomials of degree not greater than 3.
        * Error term of order `1/n**4`.
    4. Boole's rule (`degree = 4`)
        * Exact for polynomials of degree not greater than 5.
        * Error term of order `1/n**6`.
    
    Reference: [Wikipedia](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas)
    
    -----
    ### Implementation
    The closed Newton-Cotes formulas have the following form:
        `coeff_h * h * (linear_combination_of_func_values)`
    This function implements the above equation, where `coeff_h` and the coefficients of the linear combination are computed in other functions.
    """
    x = np.linspace(a, b, num_subintervals+1)
    summands = [
        integration_closed_Newton_Cotes(
            a           = x[i],
            b           = x[i+1],
            func        = func,
            func_kwargs = func_kwargs,
            degree      = degree,
        ) for i in range(num_subintervals)
    ]
    return np.sum(np.stack(summands, axis=-1), axis=-1)


def integration_romberg(
        a:              float,
        b:              float,
        func:           Callable[[np.ndarray], np.ndarray],
        func_kwargs:    Optional[dict[str, object]] = {},
        order:          int = 4,
        num_intervals:  int = 4,
    ) -> np.ndarray:
    """Numerical integration using the Romberg integration
    
    -----
    ### Description
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the Romberg integration of order `order`, a method inspired by the Richardson extrapolation and the trapezoid quadrature rule for smooth functions.
    """
    if order < 1:
        raise ValueError(f"'order' should be a positive even integer.")
    if not (isinstance(order, int) or order%2 != 0):
        warnings.warn(f"'order' should be a positive even integer. The input 'order' is rounded to the nearest even integer.", UserWarning)
        order = int(order)
        if order%2 != 0:
            order -= 1
    
    half_order: int = order // 2
    if num_intervals % (2**(half_order-1)) != 0:
        raise ValueError(f"Number of intervals should be a multiple of '2**(half_order-1)', but got {half_order=} and {num_intervals=}.")
    
    storage = [
        integration_closed_Newton_Cotes_composite(
            a=a, b=b, func=func, func_kwargs=func_kwargs, degree=1, num_subintervals=num_intervals//(2**cnt)
        ) for cnt in range(half_order)
    ]   # Length: half_order
    for i in range(1, half_order):
        for j in range(half_order-i):
            weight_plus  = float(np.power(4, i) / (np.power(4, i) - 1))
            weight_minus = float(1 - weight_plus)
            storage[j] = weight_plus * storage[j] + weight_minus * storage[j+1]
    return storage[0]
    
    
##################################################
##################################################
# 
# Deprecated functions
# 
##################################################
##################################################
def integration_romberg_backward(
        a:              float,
        b:              float,
        func:           Callable[[np.ndarray], np.ndarray],
        func_kwargs:    Optional[dict[str, object]] = {},
        order:          int = 4,
        num_intervals:  int = 4,
    ) -> np.ndarray:
    """Numerical integration using the Romberg integration
    
    -----
    (Deprecation warning) This function is deprecated and will be removed in the future. Please use `integration_romberg` instead, which is an efficient implementation of this function.
    """
    warnings.warn(
        "This function is deprecated and will be removed in the future. Please use `integration_romberg` instead, which is an efficient implementation of this function.",
        DeprecationWarning
    )
    if order < 1:
        raise ValueError(f"'order' should be a positive even integer.")
    if not (isinstance(order, int) or order%2 != 0):
        warnings.warn(f"'order' should be a positive even integer. The input 'order' is rounded to the nearest even integer.", UserWarning)
        order = int(order)
        if order%2 != 0:
            order -= 1
    
    half_order: int = order // 2
    if num_intervals % (2**(half_order-1)) != 0:
        raise ValueError(f"Number of intervals should be a multiple of '2**(half_order-1)', but got {half_order=} and {num_intervals=}.")
    
    def _romberg(_n_intervals: int, _half_order: int) -> np.ndarray:
        if _half_order == 1:
            return integration_closed_Newton_Cotes_composite(
                a=a, b=b, func=func, func_kwargs=func_kwargs, degree=1, num_subintervals=_n_intervals
            )
        else:
            if _n_intervals % 2 != 0:
                raise ValueError(f"Number of intervals should be an even number.")
            _half_order_new = _half_order - 1
            return (
                _romberg(_n_intervals, _half_order_new) +
                (_romberg(_n_intervals, _half_order_new) - _romberg(_n_intervals//2, _half_order_new)) / (np.power(4, _half_order-1) - 1)
            )
    
    return _romberg(num_intervals, half_order)


##################################################
##################################################
# End of file