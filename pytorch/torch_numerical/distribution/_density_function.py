import  torch
from    ..utils     import  EPSILON, ones, repeat


##################################################
##################################################
__all__: list[str] = [
    'maxwellian_homogeneous',
    'maxwellian_inhomogeneous',
    'bkw',
    'get_bkw_coeff_int',
    'get_bkw_coeff_ext',
]


##################################################
##################################################
def maxwellian_homogeneous(
        v:                  torch.Tensor,
        mean_density:       torch.Tensor,
        mean_velocity:      torch.Tensor,
        mean_temperature:   torch.Tensor,
    ) -> torch.Tensor:
    """Compute the local Maxwellian distribution with homogeneous input arguments.
    
    -----
    ### Note
    In this implementation, the tensors `mean_density`, `mean_velocity`, and `mean_temperature` should of shape `(B, x)` where `x==dim` for `mean_velocity` and `x==1` for the others.
    
    -----
    ### Arguments
    * `v` (`torch.Tensor`)
        The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    * `mean_density` (`torch.Tensor`)
        The mean density, which is of shape `(B, 1)`.
    * `mean_velocity` (`torch.Tensor`)
        The mean velocity, which is of shape `(B, d)`.
    * `mean_temperature` (`torch.Tensor`)
        The mean temperature, which is of shape `(B, 1)`.
    
    -----
    ### Return
    This function returns a `numpy.ndarray` object of shape `(1, *repeat(1, dim), K_1, ..., K_d, 1)`, which is the discretization of the BKW solution corresponding to the input arguments.
    Note that the spatial grid is `repeat(1, dim)`, as the BKW solution is the solution to several *homogeneous* kinetic equations.
    """
    if not (
            mean_density.shape[0] == mean_velocity.shape[0]
            and
            mean_velocity.shape[0] == mean_temperature.shape[0]
        ):
        raise ValueError(
            '\n'.join(
                (
                    f"Shape mismatch:",
                    f"* {mean_density.shape     = }",
                    f"* {mean_velocity.shape    = }",
                    f"* {mean_temperature.shape = }",
                )
            )
        )
        
    num_instances = mean_density.shape[0]
    dim = v.shape[-1]
    
    if not (mean_density.ndim == 2 and mean_density.shape[-1] == 1):
        raise ValueError(f"'mean density' should be a 2-dimensional array of shape (B, 1), but {mean_density.shape=}.")
    if not (mean_velocity.ndim == 2 and mean_velocity.shape[-1] == dim):
        raise ValueError(f"The mean velocity should be a 2-dimensional array of shape (B, dim), but {mean_velocity.shape=}.")
    if not (mean_temperature.ndim == 2 and mean_temperature.shape[-1] == 1):
        raise ValueError(f"The mean temperature should be a 2-dimensional array of shape (B, 1), but {mean_temperature.shape=}.")
    
    # Reshape the mean velocity for vectorized operations
    # NOTE (batch, dim_1, ..., dim_d, values)
    mean_density:       torch.Tensor  = \
        mean_density.reshape(    num_instances, *ones(2*dim), 1)
    mean_velocity:      torch.Tensor  = \
        mean_velocity.reshape(   num_instances, *ones(2*dim), dim)
    mean_temperature:   torch.Tensor  = \
        mean_temperature.reshape(num_instances, *ones(2*dim), 1)
    
    # Compute the Maxwellian
    _scale:     torch.Tensor  = \
        mean_density / torch.pow(2*torch.pi*mean_temperature, dim/2)
    _exp:       torch.Tensor  = \
        -torch.sum(
            (v[*repeat(None, 1+dim), ...]-mean_velocity)**2,
            dim=-1, keepdims=True
        ) / (2*mean_temperature)
    ret = _scale * torch.exp(_exp)
    
    # Return the result
    return ret


def maxwellian_inhomogeneous(
        xv:                 torch.Tensor,
        mean_density:       torch.Tensor,
        mean_velocity:      torch.Tensor,
        mean_temperature:   torch.Tensor,
        
        eps:                float   = EPSILON,
    ) -> torch.Tensor:
    """Compute the local Maxwellian distribution with inhomogeneous input arguments.
    
    -----
    ### Note
    In this implementation, the tensors `mean_density`, `mean_velocity`, and `mean_temperature` should of shape `(B, x)` where `x==dim` for `mean_velocity` and `x==1` for the others.
    
    -----
    ### Arguments
    * `xv` (`torch.Tensor`)
        The spatial-velocity grid, which is of shape `(N_1, ..., N_d, K_1, ..., K_d, 2*d)`.
    * `mean_density` (`torch.Tensor`)
        The local mean density, which is of shape `(B, N_1, ..., N_d, 1)`.
    * `mean_velocity` (`torch.Tensor`)
        The local mean velocity, which is of shape `(B, N_1, ..., N_d, d)`.
    * `mean_temperature` (`torch.Tensor`)
        The mean temperature, which is of shape `(B, N_1, ..., N_d, 1)`.
        
    -----
    ### Return
    This function returns a `numpy.ndarray` object of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`, which is the discretization of the local Maxwellian distribution corresponding to the input arguments.
    """
    if not (
            mean_density.shape[0] == mean_velocity.shape[0]
            and
            mean_velocity.shape[0] == mean_temperature.shape[0]
        ):
        raise ValueError(
            '\n'.join(
                (
                    f"Shape mismatch:",
                    f"* {mean_density.shape    =}",
                    f"* {mean_velocity.shape   =}",
                    f"* {mean_temperature.shape=}",
                )
            )
        )
        
    num_instances = mean_density.shape[0]
    dim = xv.shape[-1] // 2
    space_res = xv.shape[:dim]
    
    if not (mean_density.ndim == dim+2 and mean_density.shape[-1] == 1):
        raise ValueError(f"'mean density' should be a (dim+2)-dimensional array of shape (B, N_1, ..., N_d, 1), but {mean_density.shape=}.")
    if not (mean_velocity.ndim == dim+2 and mean_velocity.shape[-1] == dim):
        raise ValueError(f"The mean velocity should be a (dim+2)-dimensional array of shape (B, N_1, ..., N_d, dim), but {mean_velocity.shape=}.")
    if not (mean_temperature.ndim == dim+2 and mean_temperature.shape[-1] == 1):
        raise ValueError(f"The mean temperature should be a (dim+2)-dimensional array of shape (B, N_1, ..., N_d1), but {mean_temperature.shape=}.")
    
    # Reshape the mean density, velocity, temperature for vectorized operations
    # NOTE (batch, x1, ..., xd, v1, ..., vd, values)
    mean_density        = mean_density.reshape(
        num_instances, *space_res, *ones(dim), 1
    )
    mean_velocity       = mean_velocity.reshape(
        num_instances, *space_res, *ones(dim), dim
    )
    mean_temperature    = mean_temperature.reshape(
        num_instances, *space_res, *ones(dim), 1
    )
    
    # Compute the Maxwellian
    _scale  = mean_density / ((2*torch.pi*mean_temperature)**(dim/2) + eps)
    _exp    = -torch.sum((xv[..., dim:] - mean_velocity)**2, dim=-1, keepdims=True) / (2*mean_temperature + eps)
    ret = _scale * torch.exp(_exp)

    # Return the result
    return ret


##################################################
##################################################
# The BKW solution
def bkw(
        t:          torch.Tensor,
        v:          torch.Tensor,
        
        vhs_coeff:  float,  # Determines `coeff_int`
        coeff_ext:  float,
        
        scale_v:    float   = 1.0,
        
        verbose:    bool    = True,
        equation:   str     = 'boltzmann',
    ) -> torch.Tensor:
    """Returns an array of the values of the BKW solution.
    
    -----
    ### Description
    The BKW solution is a solution for the homogeneous kinetic equation (Boltzmann equation, Fokker-Planck-Landau equation) for the Maxwellian gas, i.e., the various hard sphere model with the exponent 0.
    Remark that the Maxwellian gas is the case where
    * the collision operator is defined by the various-hard-sphere (VHS) model (the collision operator, which is a function of the relative speed and the cosine displacement, is separable). and
    * the exponent of the relative speed is zero, so that the collision operator is constant.
    
    The exact formula of the BKW solution depends the following quantities:
    1. The dimension of the velocity domain.
    2. The coefficient of the exponential term (`coeff_ext`) and in the exponential coefficient in the term (`coeff_int`).
        * `coeff_int` can be recovered from the collision operator, if it is not given.
        * `coeff_ext` should be given manually.
    
    -----
    ### Return
    This function returns a `numpy.ndarray` object of shape
        `(1, num_timestamps, *repeat(1, dim), K_1, ..., K_d, 1)`,
    which is the discretization of the BKW solution corresponding to the input arguments.
    Note that the spatial grid is `repeat(1, dim)`, as the BKW solution is the solution to several *homogeneous* kinetic equations.
    
    -----
    ### Note
    Currently, the vectorized implementation is not supported.
    """
    equation = equation.lower()
    
    dim = v.shape[-1]
    coeff_int = get_bkw_coeff_int(dim, vhs_coeff, equation)
    
    _line   = '-'*10
    _front  = ''.join((_line, "[ BKW solution ]", _line))
    _back   = '-' * len(_front)
    if verbose:
        print(_front)
        print(f"* coeff_ext: {coeff_ext}")
        print(f"* coeff_int: {coeff_int}")
        print(_back)
    
    # Dimension alignment: (batch, time, *displacement, *velocity, values)
    _t:         torch.Tensor = \
        t.reshape(1, -1, *ones(dim), *ones(dim), 1)
    _speed_sq:  torch.Tensor = \
        torch.reshape(torch.sum((v/scale_v)**2, dim=-1), (1, 1, *repeat(1, dim), *v.shape[:-1], 1))
    K_t:        torch.Tensor = \
        1 - coeff_ext * torch.exp(-(scale_v**dim) * coeff_int * _t)
    
    _part1: torch.Tensor = torch.pow(2 * torch.pi * K_t * (scale_v**2), -dim/2)
    _part2: torch.Tensor = torch.exp(-_speed_sq / (2*K_t))
    _part3: torch.Tensor = ((dim+2)/2) - ((dim+_speed_sq)/2) / K_t + (_speed_sq/2) / (K_t**2)
    
    ret = _part1 * _part2 * _part3
    return ret


def get_bkw_coeff_int(dim: int, vhs_coeff: float, equation: str) -> float:
    """Computes the coefficient of time in the exponential term, which comprises the BKW solution.
    
    -----
    The interior coefficient (the coefficient in the exponent) is computed as `scale * vhs_coeff`, where the values of `scale` are listed below.
    
    1. Boltzmann equation (`equation` is in `["boltzmann"]`)
        * `scale = 1 / 2` if `dim==1`.
        * `scale = 1 * pi / 4` if `dim==2`.
        * `scale = 2 * pi / 3` if `dim==3`.
    
    2. Fokker-Planck-Landau equation (`equation` is in `["fpl", "fokker-planck-landau"]`)
        * `scale = 2*(dim-1)` for all values of `dim`.
    """
    if equation in ['boltzmann']:
        if dim==2:
            return (torch.pi/4) * vhs_coeff
        elif dim==3:
            return (2*torch.pi/3) * vhs_coeff
        else:
            raise NotImplementedError(f"This function is implemented only for 2D or 3D Maxwellian gas. (dim={dim})")
    if equation in ['fpl', "fokker-planck-landau"]:
        return 2*(dim-1)*vhs_coeff


def get_bkw_coeff_ext(dim: int) -> float:
    if dim==2:
        return 0.5
    elif dim==3:
        return 1.0
    else:
        raise NotImplementedError(f"Check the dimension. ({dim=})")


##################################################
##################################################
# End of file