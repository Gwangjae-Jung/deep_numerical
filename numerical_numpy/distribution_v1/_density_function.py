from    typing      import  Optional

import  numpy       as      np
import  torch

from    ..utils         import  EPSILON, ArrayData, ones, repeat


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
        v:                  np.ndarray,
        mean_density:       np.ndarray,
        mean_velocity:      np.ndarray,
        mean_temperature:   np.ndarray,
        
        as_numpy:           bool = True,
    ) -> ArrayData:
    """Compute the local Maxwellian distribution with homogeneous input arguments.
    
    -----
    ### Note
    In this implementation, the tensors `mean_density`, `mean_velocity`, and `mean_temperature` should of shape `(B, x)` where `x==dim` for `mean_velocity` and `x==1` for the others.
    
    -----
    ### Arguments
    * `v` (`np.ndarray`)
        The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    * `mean_density` (`np.ndarray`)
        The mean density, which is of shape `(B, 1)`.
    * `mean_velocity` (`np.ndarray`)
        The mean velocity, which is of shape `(B, d)`.
    * `mean_temperature` (`np.ndarray`)
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
    mean_density:       np.ndarray  = \
        mean_density.reshape(    num_instances, *ones(2*dim), 1)
    mean_velocity:      np.ndarray  = \
        mean_velocity.reshape(   num_instances, *ones(2*dim), dim)
    mean_temperature:   np.ndarray  = \
        mean_temperature.reshape(num_instances, *ones(2*dim), 1)
    
    # Compute the Maxwellian
    _scale:     np.ndarray  = \
        mean_density / np.power(2*np.pi*mean_temperature, dim/2)
    _exp:       np.ndarray  = \
        -np.sum(
            (v[*repeat(None, 1+dim), ...]-mean_velocity)**2,
            axis=-1, keepdims=True
        ) / (2*mean_temperature)
    ret = _scale * np.exp(_exp)
    
    # Reshape and return the result
    ret = ret.squeeze(-1)
    if not as_numpy:
        ret = torch.tensor(ret, dtype=torch.float)
    return ret


def maxwellian_inhomogeneous(
        xv:                 np.ndarray,
        mean_density:       np.ndarray,
        mean_velocity:      np.ndarray,
        mean_temperature:   np.ndarray,
        
        as_numpy:           bool    = True,
        
        eps:                float   = EPSILON,
    ) -> ArrayData:
    """Compute the local Maxwellian distribution with inhomogeneous input arguments.
    
    -----
    ### Note
    In this implementation, the tensors `mean_density`, `mean_velocity`, and `mean_temperature` should of shape `(B, x)` where `x==dim` for `mean_velocity` and `x==1` for the others.
    
    -----
    ### Arguments
    * `xv` (`np.ndarray`)
        The spatial-velocity grid, which is of shape `(N_1, ..., N_d, K_1, ..., K_d, 2*d)`.
    * `mean_density` (`np.ndarray`)
        The local mean density, which is of shape `(B, N_1, ..., N_d, 1)`.
    * `mean_velocity` (`np.ndarray`)
        The local mean velocity, which is of shape `(B, N_1, ..., N_d, d)`.
    * `mean_temperature` (`np.ndarray`)
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
    _scale  = mean_density / (np.power(2*np.pi*mean_temperature, dim/2) + eps)
    _exp    = -np.sum((xv[..., dim:] - mean_velocity)**2, axis=-1, keepdims=True) / (2*mean_temperature + eps)
    ret = _scale * np.exp(_exp)

    # Return the result
    if not as_numpy:
        ret = torch.tensor(ret, dtype=torch.float)
    return ret


##################################################
##################################################
# The BKW solution
def bkw(
        t:          np.ndarray,
        v:          np.ndarray,
        
        vhs_coeff:  float,  # Determines `coeff_int`
        coeff_ext:  float,
        
        scale_v:    float   = 1.0,
        as_numpy:   bool    = True,
        
        verbose:    bool    = True,
        equation:   str     = 'boltzmann',
    ) -> ArrayData:
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
        `(num_timestamps, K_1, ..., K_d, 1)`,
    which is the discretization of the BKW solution corresponding to the input arguments.
    Note that the spatial grid is `repeat(1, dim)`, as the BKW solution is the solution to several *homogeneous* kinetic equations.
    
    -----
    ### Note
    Currently, the vectorized implementation is not supported.
    """
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
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
    
    # Dimension alignment: (batch, time, *displacement, *velocity)
    _t:         np.ndarray = \
        t.reshape(-1, *ones(dim))
    _speed_sq:  np.ndarray = \
        np.reshape(np.sum((v/scale_v)**2, axis=-1), (1, *v.shape[:-1]))
    K_t:        np.ndarray = \
        1 - coeff_ext * np.exp(-(scale_v**dim) * coeff_int * _t)
    
    _part1: np.ndarray = np.power(2 * np.pi * K_t * (scale_v**2), -dim/2)
    _part2: np.ndarray = np.exp(-_speed_sq / (2*K_t))
    _part3: np.ndarray = ((dim+2)/2) - ((dim+_speed_sq)/2) / K_t + (_speed_sq/2) / (K_t**2)
    
    ret = _part1 * _part2 * _part3
    if as_numpy:
        return ret
    else:
        return torch.tensor(ret, dtype=torch.float)


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
            return (np.pi/4) * vhs_coeff
        elif dim==3:
            return (2*np.pi/3) * vhs_coeff
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