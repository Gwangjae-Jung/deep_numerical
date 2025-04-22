import  numpy           as      np
from    ..utils         import  EPSILON, zeros, ones, repeat


##################################################
##################################################
__all__: list[str] = [
    'compute_moments_homogeneous',
    'compute_moments_inhomogeneous',
    'compute_energy_homogeneous',
    'compute_energy_inhomogeneous',
    'compute_entropy_homogeneous',
    'compute_entropy_inhomogeneous',
]


##################################################
##################################################
# Parameters of the Maxwellian distribution
def compute_moments_homogeneous(
        f:      np.ndarray,
        v:      np.ndarray,
        eps:    float       = EPSILON,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the physical quantities which determine the local Maxwellian distribution - mass, velocity, and temperature.
    
    -----
    ### Arguments
    * `f` (`numpy.ndarray`)
        The distribution function at a specific time, which is of shape `(B, *repeat(1, dim), K_1, ..., K_d, 1)`.
    * `v` (`numpy.ndarray`)
       The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    * `eps` (`float`, default: `_EPSILON`)
        The value which is used to prevent divisions by zero.
    
    -----
    ### Return
    This function returns the tuple `(rho, u, T)`, where `rho`, `u`, and `T` are `numpy.ndarray` objects described below.
    * `rho` saves the mean density of each instance, which is of shape `(B, 1)`.
    * `u` saves the mean velocity of each instance, which is of shape `(B, d)`.
    * `T` saves the mean temperature of each instance, which is of shape `(B, 1)`.
    """    
    # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv  = float(np.prod(v[ones(dim)] - v[zeros(dim)]))
    
    # Reshape `v` in this function for vectorized operations
    v = v[*repeat(None, 1+dim), ...]    # Shape: (1, *repeat(1, dim), K_1, ..., K_d, d)
    v_axes = tuple(range(1+dim, 1+2*dim)) # The following `dim` dimensions
    
    # Compute the density
    density: np.ndarray = f.sum(axis=v_axes, keepdims=True) * dv
    
    # Compute the average velocity
    momentum: np.ndarray = np.sum(f*v, axis=v_axes, keepdims=True) * dv
    velocity: np.ndarray = momentum / (density + eps)
    
    # Compute the temperature
    _speed_sq:   np.ndarray = np.sum((v-velocity)**2, axis=-1, keepdims=True)
    temperature: np.ndarray = np.sum(f*_speed_sq, axis=v_axes, keepdims=True) * dv / (dim*density + eps)
    
    # Return
    return (density, velocity, temperature)


def compute_moments_inhomogeneous(
        f:      np.ndarray,
        v:      np.ndarray,
        eps:    float       = EPSILON,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the physical quantities which determine the local Maxwellian distribution - mass, velocity, and temperature.
    
    -----
    ### Arguments
    * `f` (`numpy.ndarray`)
        The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    * `v` (`numpy.ndarray`)
        * The velocity grid.
        The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    * `eps` (`float`, default: `_EPSILON`)
        The value which is used to prevent divisions by zero.

    -----
    ### Return
    This function returns the tuple `(rho, u, T)`, where `rho`, `u`, and `T` are `numpy.ndarray` objects described below.
    * `rho` saves the mean density of each instance, which is of shape `(B, N_1, ..., N_d, 1)`.
    * `u` saves the mean velocity of each instance, which is of shape `(B, N_1, ..., N_d, d)`.
    * `T` saves the mean temperature of each instance, which is of shape `(B, N_1, ..., N_d, 1)`.
    
    Unlike the global mean density and temperature as the output of the function `compute_moments_homogeneous(...)` are of type `float`, `rho` and `T` are treated as tensors of the same dimension as `u` so that the output quantities can be readily used as input arguments of the function `maxwellian_inhomogeneous(...)`. See the following example.
    """
    # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv  = float(np.prod(v[ones(dim)] - v[zeros(dim)]))
    
    # Reshape the velocity grid for vectorized operations
    v = v.reshape(1, *ones(dim), *v.shape)
    v_axes = tuple((range(1+dim, 1+2*dim)))
    """The tuple of the velocity dimensions, where the last dimension of each tensor in this function is assumed to save values."""
    
    # Compute the density
    density: np.ndarray = f.sum(axis=v_axes, keepdims=True) * dv
    
    # Compute the average velocity
    momentum: np.ndarray = np.sum(f*v, axis=v_axes, keepdims=True) * dv
    velocity: np.ndarray = momentum / (density + eps)
    
    # Compute the temperature
    _speed_sq:   np.ndarray = np.sum((v - velocity)**2, axis=-1, keepdims=True)
    temperature: np.ndarray = np.sum(f*_speed_sq, axis=v_axes, keepdims=True) * dv / (dim*density + eps)
        
    # Return
    density     = density.squeeze(v_axes)
    velocity    = velocity.squeeze(v_axes)
    temperature = temperature.squeeze(v_axes)
    return (density, velocity, temperature)


##################################################
##################################################
# Energy
def compute_energy_homogeneous(
        f:  np.ndarray,
        v:  np.ndarray,
    ) -> np.ndarray:
    """Computes the kinetic energy (the velocity integral of `f * (abs(v)**2) / 2`).
    
    -----
    ### Arguments
    * `f` (`numpy.ndarray`)
        The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
    * `v` (`numpy.ndarray`)
        The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    """
    # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv = np.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, v1, ..., vd, value)
    v = v[None, ...]
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute
    speed_sq = np.sum(v**2, axis=-1, keepdims=True)
    energy = np.sum(f*speed_sq, axis=axes_v, keepdims=True) * dv / 2
    energy = energy.squeeze(axis=axes_v)
    return energy


def compute_energy_inhomogeneous(
        f:  np.ndarray,
        v:  np.ndarray,
        dx: float,
    ) -> np.ndarray:
    """Computes the kinetic energy (the velocity integral of `f * (abs(v)**2) / 2`).
    
    -----
    ### Arguments
    * `f` (`numpy.ndarray`)
        The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    * `v` (`numpy.ndarray`)
        The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    * `dx` (`float`)
        The spatial grid size.
    """
    # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv  = np.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, x1, ..., xd, v1, ..., vd, value)
    v = v.reshape(1, *ones(dim), *v.shape)
    axes_x = tuple((+(1+k) for k in range(dim)))
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute
    speed_sq = np.sum(v**2, axis=-1, keepdims=True)
    energy = np.sum(f*speed_sq, axis=(*axes_x, *axes_v), keepdims=True) * dv * dx / 2
    energy = np.squeeze(energy, axis=(*axes_x, *axes_v))
    return energy


##################################################
##################################################
# Entropy
def compute_entropy_homogeneous(
        f:      np.ndarray,
        v:      np.ndarray,
        eps:    float = EPSILON,
    ) -> np.ndarray:
    # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv = np.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, v1, ..., vd, value)
    f = f[..., None]
    v = v[None, ...]
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute
    _integrand: np.ndarray
    if float(f.min()) + eps > 0:
        _integrand = f * np.log(f)
    else:
        _integrand = f * np.log(f + eps)
    
    return np.sum(_integrand, axis=axes_v) * dv


def compute_entropy_inhomogeneous(
        f:      np.ndarray,
        v:      np.ndarray,
        dx:     float,
        eps:    float = EPSILON,
    ) -> np.ndarray:
        # Retrieve the dimension and dv
    dim = v.shape[-1]
    dv  = np.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, x1, ..., xd, v1, ..., vd, value)
    f = f[..., None]
    v = v.reshape(1, *ones(dim), *v.shape)
    axes_x = tuple((+(1+k) for k in range(dim)))
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute
    _integrand: np.ndarray
    if float(f.min()) + eps > 0:
        _integrand = f * np.log(f)
    else:
        _integrand = f * np.log(f + eps)
    
    return np.sum(_integrand, axis=(*axes_x, *axes_v)) * dx * dv


##################################################
##################################################
# End of file