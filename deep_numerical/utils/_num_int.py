"""
### Note
All numerical integrators in this script assume that the numeric input `r` of `func` (the input except for those in `func_kwargs`) is a 1-dimensional array.
To conduct numerical integration, a reshape of `r` should be implemented inside `func`.
"""
import  warnings
import  torch
from    typing          import  Callable, Optional

from    ._dtype         import  TORCH_DEFAULT_DTYPE, TORCH_DEFAULT_DEVICE
from    ._quadrature    import  DEFAULT_QUAD_ORDER_LEBEDEV, roots_legendre_shifted, roots_lebedev


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
    'integration_romberg',
]


##################################################
##################################################
# Numerical integration
def integration_guass_legendre(
        num_roots:      int,
        a:              float,
        b:              float,
        func:           Callable[[torch.Tensor, object], torch.Tensor],
        func_kwargs:    Optional[dict[str, object]] = {},
        dtype:          torch.dtype     = TORCH_DEFAULT_DTYPE,
        device:         torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """
    ## Numerical integration on a compact interval using the Gauss-Legendre quadrature rule
    
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the Gauss-Legendre quadrature rule of order `num_roots`.
    
    Arguments:
        `num_roots` (`int`): The number of roots of the Gauss-Legendre quadrature rule.
        `a` (`float`): The lower bound of the interval.
        `b` (`float`): The upper bound of the interval.
        `func` (`Callable[[torch.Tensor, object], torch.Tensor]`):  The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `func_kwargs` (`Optional[dict[str, object]]`): The other arguments of `func`.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    """
    roots, weights = roots_legendre_shifted(num_roots, a, b, dtype=dtype, device=device)
    func_vals = func(roots, **func_kwargs)
    return torch.einsum("...t,t->...", func_vals, weights)


def integration_lebedev(
        f:                  Callable[[torch.Tensor], float],
        quad_order_lebedev: int = DEFAULT_QUAD_ORDER_LEBEDEV,
        dtype:              torch.dtype     = TORCH_DEFAULT_DTYPE,
        device:             torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """
    ## Numerical integration on S2 using the Lebedev quadrature rule
    
    This function supports the numerical integration of a complex-valued function `f`, using the Lebedev quadrature rule of order `quad_order_lebedev`.
    
    Currently, the parallel computation is not implemented, which is easy to be implemented.
    
    Arguments:
        `f` (`Callable[[torch.Tensor], float]`): The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `quad_order_lebedev` (`int`): The order of the Lebedev quadrature rule.
        `dtype` (`torch.dtype`): The data type of the input tensor.
        `device` (`torch.device`): The device on which the input tensor is located.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    """
    roots, weights = roots_lebedev(quad_order_lebedev, dtype=dtype, device=device)
    return torch.sum(f(roots) * weights)


integration_legendre    = integration_guass_legendre
integration_S2          = integration_lebedev


##################################################
##################################################
def integration_closed_Newton_Cotes(
        a:              float,
        b:              float,
        func:           Callable[[torch.Tensor], torch.Tensor],
        func_kwargs:    Optional[dict[str, object]] = {},
        degree:         int = 4,
    ) -> torch.Tensor:
    """
    ## Numerical integration using the closed Newton-Cotes formula
    
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
    
    Arguments:
        `a` (`float`): The lower bound of the interval.
        `b` (`float`): The upper bound of the interval.
        `func` (`Callable[[torch.Tensor], torch.Tensor]`):  The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `func_kwargs` (`Optional[dict[str, object]]`): The other arguments of `func`.
        `degree` (`int`): The degree of the closed Newton-Cotes formula.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    
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
    grid_points = torch.linspace(a, b, degree+1, dtype=torch.float64)
    step_size = float((b-a) / degree)
    
    coeff_h, coeff_LC = _closed_Newton_Cotes_coefficients(degree)
    linear_comb = torch.einsum("i,...i->...", coeff_LC, func(grid_points, **func_kwargs))
    return coeff_h * step_size * linear_comb


def _closed_Newton_Cotes_coefficients(degree: int) -> tuple[float, torch.Tensor]:
    if degree == 1:
        return (0.5, torch.tensor([1, 1]))
    elif degree == 2:
        return (1/3, torch.tensor([1, 4, 1]))
    elif degree == 3:
        return (3/8, torch.tensor([1, 3, 3, 1]))
    elif degree == 4:
        return (2/45, torch.tensor([7, 32, 12, 32, 7]))
    else:
        raise NotImplementedError(f"Unsupported degree: {degree}.")


def integration_open_Newton_Cotes(
        a:              float,
        b:              float,
        func:           Callable[[torch.Tensor], torch.Tensor],
        func_kwargs:    Optional[dict[str, object]] = {},
        degree:         int = 3,
    ) -> torch.Tensor:
    """
    ## Numerical integration using the open Newton-Cotes formula
    
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
    
    Arguments:
        `a` (`float`): The lower bound of the interval.
        `b` (`float`): The upper bound of the interval.
        `func` (`Callable[[torch.Tensor], torch.Tensor]`):  The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `func_kwargs` (`Optional[dict[str, object]]`): The other arguments of `func`.
        `degree` (`int`): The degree of the open Newton-Cotes formula.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
    
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
    grid_points = torch.linspace(a, b, degree+3, dtype=torch.float64)[1:-1]
    step_size = float((b-a) / (degree+2))
    
    coeff_h, coeff_LC = _open_Newton_Cotes_coefficients(degree)
    linear_comb = torch.einsum("i,...i->...", coeff_LC, func(grid_points, **func_kwargs))
    return coeff_h * step_size * linear_comb


def _open_Newton_Cotes_coefficients(degree: int) -> tuple[float, torch.Tensor]:
    if degree == 0:
        return (2, torch.tensor([1]))
    elif degree == 1:
        return (3/2, torch.tensor([1, 1]))
    elif degree == 2:
        return (4/3, torch.tensor([2, -1, 2]))
    elif degree == 3:
        return (5/24, torch.tensor([11, 1, 1, 11]))
    else:
        raise ValueError(f"Unsupported degree: {degree}.")



def integration_closed_Newton_Cotes_composite(
        a:                  float,
        b:                  float,
        func:               Callable[[torch.Tensor], torch.Tensor],
        func_kwargs:        Optional[dict[str, object]] = {},
        degree:             int = 4,
        num_subintervals:   int = 2,
    ) -> torch.Tensor:
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
    x = torch.linspace(a, b, num_subintervals+1)
    summands = [
        integration_closed_Newton_Cotes(
            a           = x[i],
            b           = x[i+1],
            func        = func,
            func_kwargs = func_kwargs,
            degree      = degree,
        ) for i in range(num_subintervals)
    ]
    return torch.sum(torch.stack(summands, dim=-1), dim=-1)


def integration_romberg(
        a:              float,
        b:              float,
        func:           Callable[[torch.Tensor], torch.Tensor],
        func_kwargs:    Optional[dict[str, object]] = {},
        order:          int = 4,
        num_intervals:  int = 4,
    ) -> torch.Tensor:
    """
    ## Numerical integration using the Romberg integration
    
    This function supports the numerical integration of a tensor-valued function `func` on a compact interval `[a, b]`, using the Romberg integration of order `order`, a method inspired by the Richardson extrapolation and the trapezoid quadrature rule for smooth functions.
    
    Arguments:
        `a` (`float`): The lower bound of the interval.
        `b` (`float`): The upper bound of the interval.
        `func` (`Callable[[torch.Tensor], torch.Tensor]`):  The integrand. The numeric input `r` of `func` should be a 1-dimensional tensor, and it should be implemented so that the summands are aligned in the last dimension.
        `func_kwargs` (`Optional[dict[str, object]]`): The other arguments of `func`.
        `order` (`int`): The order of the Romberg integration.
        `num_intervals` (`int`): The number of intervals for the trapezoid rule.
    
    Returns:
        `torch.Tensor`: The result of the numerical integration.
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
            weight_plus  = float(torch.power(4, i) / (torch.power(4, i) - 1))
            weight_minus = float(1 - weight_plus)
            storage[j] = weight_plus * storage[j] + weight_minus * storage[j+1]
    return storage[0]


##################################################
##################################################
# End of file