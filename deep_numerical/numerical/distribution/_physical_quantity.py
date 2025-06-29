from    typing      import  Optional, Sequence
import  torch
from    ...utils    import  EPSILON, zeros, ones, repeat


##################################################
##################################################
__all__: list[str] = [
    'compute_moments_homogeneous',
    'compute_moments_inhomogeneous',
    'compute_mass_homogeneous',
    'compute_mass_inhomogeneous',
    'compute_momentum_homogeneous',
    'compute_momentum_inhomogeneous',
    'compute_energy_homogeneous',
    'compute_energy_inhomogeneous',
    'compute_entropy_homogeneous',
    'compute_entropy_inhomogeneous',
    
    'plot_quantities_homogeneous',
]


##################################################
##################################################
# Parameters of the Maxwellian distribution
def compute_moments_homogeneous(
        f:      torch.Tensor,
        v:      torch.Tensor,
        eps:    float       = EPSILON,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the physical quantities which determine the local Maxwellian distribution - mass, velocity, and temperature.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, *repeat(1, dim), K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `eps` (`float`, default: `_EPSILON`):
            The value which is used to prevent divisions by zero.
    
    Returns:
        This function returns the tuple `(rho, u, T)`, where `rho`, `u`, and `T` are `torch.Tensor` objects described below.
            * `rho` saves the mean density of each instance, which is of shape `(B, 1)`.
            * `u` saves the mean velocity of each instance, which is of shape `(B, d)`.
            * `T` saves the mean temperature of each instance, which is of shape `(B, 1)`.
    """    
    # Retrieve the dimension and `dV`
    dim = v.shape[-1]
    dV  = float(torch.prod(v[ones(dim)] - v[zeros(dim)]))
    
    # Reshape `v` in this function for vectorized operations
    v = v[*repeat(None, 1+dim), ...]    # Shape: (1, *repeat(1, dim), K_1, ..., K_d, d)
    v_axes = tuple(range(1+dim, 1+2*dim)) # The following `dim` dimensions
    
    # Compute the density
    density: torch.Tensor = f.sum(axis=v_axes, keepdims=True) * dV
    
    # Compute the average velocity
    momentum: torch.Tensor = torch.sum(f*v, axis=v_axes, keepdims=True) * dV
    velocity: torch.Tensor = momentum / (density + eps)
    
    # Compute the temperature
    _speed_sq:   torch.Tensor = torch.sum((v-velocity)**2, axis=-1, keepdims=True)
    temperature: torch.Tensor = torch.sum(f*_speed_sq, axis=v_axes, keepdims=True) * dV / (dim*density + eps)
    
    # Squeeze the dimensions (`...xvc->...c`) and return
    dim_squeezed = tuple((-(2+k) for k in range(2*dim)))
    density     = density.squeeze(dim_squeezed)
    velocity    = velocity.squeeze(dim_squeezed)
    temperature = temperature.squeeze(dim_squeezed)
    return (density, velocity, temperature)


def compute_moments_inhomogeneous(
        f:      torch.Tensor,
        v:      torch.Tensor,
        eps:    float       = EPSILON,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the physical quantities which determine the local Maxwellian distribution - mass, velocity, and temperature.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `eps` (`float`, default: `_EPSILON`):
            The value which is used to prevent divisions by zero.

    Returns:
        This function returns the tuple `(rho, u, T)`, where `rho`, `u`, and `T` are `torch.Tensor` objects described below.
            * `rho` saves the mean density of each instance, which is of shape `(B, N_1, ..., N_d, 1)`.
            * `u` saves the mean velocity of each instance, which is of shape `(B, N_1, ..., N_d, d)`.
            * `T` saves the mean temperature of each instance, which is of shape `(B, N_1, ..., N_d, 1)`.
    """
    # Retrieve the dimension and `dV`
    dim = v.shape[-1]
    dV  = float(torch.prod(v[ones(dim)] - v[zeros(dim)]))
    
    # Reshape the velocity grid for vectorized operations
    v = v.reshape(1, *ones(dim), *v.shape)
    v_axes = tuple((range(1+dim, 1+2*dim)))
    """The tuple of the velocity dimensions, where the last dimension of each tensor in this function is assumed to save values."""
    
    # Compute the density
    density: torch.Tensor = f.sum(axis=v_axes, keepdims=True) * dV
    
    # Compute the average velocity
    momentum: torch.Tensor = torch.sum(f*v, axis=v_axes, keepdims=True) * dV
    velocity: torch.Tensor = momentum / (density + eps)
    
    # Compute the temperature
    _speed_sq:   torch.Tensor = torch.sum((v - velocity)**2, axis=-1, keepdims=True)
    temperature: torch.Tensor = torch.sum(f*_speed_sq, axis=v_axes, keepdims=True) * dV / (dim*density + eps)
        
    # Squeeze the dimensions (`...xvc->...xc`) and return
    density     = density.squeeze(v_axes)
    velocity    = velocity.squeeze(v_axes)
    temperature = temperature.squeeze(v_axes)
    return (density, velocity, temperature)


##################################################
##################################################
# Mass
def compute_mass_homogeneous(
        f:      torch.Tensor,
        dv:     float,
        dim:    Optional[int] = None,
    ) -> torch.Tensor:
    """Computes the mass (the velocity integral of `f`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
        `dv` (`float`):
            The size of the velocity grid.
        `dim` (`Optioal[int]`, default: `None`):
            The dimension of the velocity space. If `None`, `dim` is set `(f.ndim-2)//2`.
    
    Returns:
        This function returns the mass of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension and `dV`
    if dim is None:
        dim = (f.ndim-2)//2
    dV = dv**dim
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the momentum
    mass = torch.sum(f, dim=axes_v) * dV
    mass = torch.squeeze(mass, dim=axes_v)
    return mass


def compute_mass_inhomogeneous(
        f:      torch.Tensor,
        dx:     float,
        dv:     float,
        dim:    Optional[int] = None,
    ) -> torch.Tensor:
    """Computes the momentum (the space-velocity integral of `f * v`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
        `dx` (`float`):
            The spatial grid size.
        `dv` (`float`):
            The size of the velocity grid.
        `dim` (`Optioal[int]`, default: `None`):
            The dimension of the velocity space. If `None`, `dim` is set `(f.ndim-2)//2`.

    Returns:
        This function returns the mass of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension, `dX`, and `dV`
    if dim is None:
        dim = (f.ndim-2)//2
    dX = dx**dim
    dV = dv**dim
    axes_x = tuple((+(1+k) for k in range(dim)))
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the momentum
    mass = torch.sum(f, dim=(*axes_x, *axes_v)) * (dX*dV)
    return mass


##################################################
##################################################
# Momentum
def compute_momentum_homogeneous(
        f:  torch.Tensor,
        v:  torch.Tensor,
    ) -> torch.Tensor:
    """Computes the momentum (the velocity integral of `f * v`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    
    Returns:
        This function returns the momentum of each instance, which is of shape `(B, d)`.
    """
    # Retrieve the dimension and `dV`
    dim = v.shape[-1]
    dV = torch.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, v1, ..., vd, value)
    v = v[None, ...]
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the momentum
    momentum = torch.sum(f*v, dim=axes_v) * dV
    momentum = torch.squeeze(momentum, dim=axes_v)
    return momentum


def compute_momentum_inhomogeneous(
        f:  torch.Tensor,
        v:  torch.Tensor,
        dx: float,
    ) -> torch.Tensor:
    """Computes the momentum (the space-velocity integral of `f * v`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `dx` (`float`):
            The spatial grid size.

    Returns:
        This function returns the momentum of each instance, which is of shape `(B, d)`.
    """
    # Retrieve the dimension, `dX`, and `dV`
    dim = v.shape[-1]
    dX = dx**dim
    dV  = torch.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, x1, ..., xd, v1, ..., vd, value)
    v = v.reshape(1, *ones(dim), *v.shape)
    axes_x = tuple((+(1+k) for k in range(dim)))
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the momentum
    momentum = torch.sum(f*v, dim=(*axes_x, *axes_v), keepdims=True) * (dX*dV)
    momentum = torch.squeeze(momentum, dim=(*axes_x, *axes_v))
    return momentum


##################################################
##################################################
# Energy
def compute_energy_homogeneous(
        f:  torch.Tensor,
        v:  torch.Tensor,
    ) -> torch.Tensor:
    """Computes the kinetic energy (the velocity integral of `f * (abs(v)**2) / 2)`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
    
    Returns:
        This function returns the kinetic energy of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension and `dV`
    dim = v.shape[-1]
    dV = torch.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, v1, ..., vd, value)
    v = v[None, ...]
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the energy
    speed_sq = torch.sum(v**2, dim=-1, keepdims=True)
    energy = torch.sum(f*speed_sq, dim=axes_v) * dV / 2
    energy = torch.squeeze(energy, dim=axes_v)
    return energy


def compute_energy_inhomogeneous(
        f:  torch.Tensor,
        v:  torch.Tensor,
        dx: float,
    ) -> torch.Tensor:
    """Computes the kinetic energy (the space-velocity integral of `f * (abs(v)**2) / 2)`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `dx` (`float`):
            The spatial grid size.

    Returns:
        This function returns the kinetic energy of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension, `dX`, and `dV`
    dim = v.shape[-1]
    dX = dx**dim
    dV = torch.prod(v[ones(dim)] - v[zeros(dim)])
    # Reshape `f` and `v` in this function for vectorized operations
    # NOTE: (batch, x1, ..., xd, v1, ..., vd, value)
    v = v.reshape(1, *ones(dim), *v.shape)
    axes_x = tuple((+(1+k) for k in range(dim)))
    axes_v = tuple((-(2+k) for k in range(dim)))
    # Compute the energy
    speed_sq = torch.sum(v**2, dim=-1, keepdims=True)
    energy = torch.sum(f*speed_sq, dim=(*axes_x, *axes_v), keepdims=True) * (dX*dV) / 2
    energy = torch.squeeze(energy, dim=(*axes_x, *axes_v))
    return energy


##################################################
##################################################
# Entropy
def compute_entropy_homogeneous(
        f:      torch.Tensor,
        dv:     torch.Tensor,
        dim:    Optional[int]   = None,
        eps:    float           = EPSILON,
    ) -> torch.Tensor:
    """Computes the kinetic energy (the velocity integral of `f * log(f))`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, *ones(d), K_1, ..., K_d, 1)`.
        `dv` (`float`):
            The size of the velocity grid. Do not be confused with the volume of the cells in the velocity grid.
        `dim` (`Optioal[int]`, default: `None`):
            The dimension of the velocity space. If `None`, `dim` is set `(f.ndim-2)//2`.
        `eps` (`float`, default: `_EPSILON`):
            The value which is used to prevent the computation of the logarithm of non-positive numbers.
    
    Returns:
        This function returns the entropy of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension and `dV`
    if dim is None:
        dim = (f.ndim-2)//2
    axes_v = tuple((-(2+k) for k in range(dim)))
    dV = dv**dim
    # Compute the entropy
    if f.min() <= 0:
        f = f - f.min() + eps
    entropy = torch.sum(f * f.log(), dim=axes_v) * dV
    entropy = torch.squeeze(entropy, dim=axes_v)
    return entropy


def compute_entropy_inhomogeneous(
        f:      torch.Tensor,
        dx:     float,
        dv:     float,
        dim:    Optional[int]   = None,
        eps:    float           = EPSILON,
    ) -> torch.Tensor:
    """Computes the kinetic energy (the space-velocity integral of `f * log(f))`.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
        `dx` (`float`):
            The size of the space grid.
        `dv` (`float`):
            The size of the velocity grid.
        `dim` (`Optioal[int]`, default: `None`):
            The dimension of the velocity space. If `None`, `dim` is set `(f.ndim-2)//2`.
        `eps` (`float`, default: `_EPSILON`):
            The value which is used to prevent the computation of the logarithm of non-positive numbers.
    
    Returns:
        This function returns the entropy of each instance, which is of shape `(B, 1)`.
    """
    # Retrieve the dimension, `dX`, and `dV`
    if dim is None:
        dim = (f.ndim-2)//2
    axes_x = tuple((-(2+k+dim)  for k in range(dim)))
    axes_v = tuple((-(2+k)      for k in range(dim)))
    dX = dx**dim
    dV = dv**dim
    # Compute the entropy
    if f.min() <= 0:
        f = f - f.min() + eps
    entropy = torch.sum(f * f.log(), dim=(*axes_x, *axes_v), keepdims=True) * (dX*dV)
    entropy = torch.squeeze(entropy, dim=(*axes_x, *axes_v))
    return entropy


##################################################
##################################################
# Plot
from    matplotlib.axes     import Axes
from    matplotlib.figure   import Figure
def plot_quantities_homogeneous(
        arr_f:  torch.Tensor,
        v_grid: torch.Tensor,
        arr_t:  Optional[torch.Tensor]  = None,
        dim:    Optional[int]           = None,
        eps:    float                   = EPSILON,
        ##############################
        figsize: tuple[int, int]    = (10, 7),
        dpi:     int                = 100,
        mode:    str                = 'plot',
        suptitle_fontsize:  int     = 20,
        title_fontsize:     int     = 12,
        plot_linewidth:     float   = 1.0,
        scatter_size:       float   = 10.0,
    ) -> tuple[Figure, Sequence[Axes]]:
    mode = mode.lower()
    assert mode in ('plot', 'scatter'), f"Invalid mode: {mode}. Choose 'plot' or 'scatter'."
    from    itertools           import  product
    import  matplotlib.pyplot   as      plt
    
    if dim is None:
        dim = (arr_f.ndim-2)//2
    if arr_t is None:
        arr_t = torch.arange(arr_f.shape[0], dtype=torch.long, device=arr_f.device)
    dv = (v_grid[ones(dim)] - v_grid[zeros(dim)])[0].item()
    mass        = compute_mass_homogeneous(arr_f, dv=dv, dim=dim)
    momentum    = compute_momentum_homogeneous(arr_f, v_grid)
    energy      = compute_energy_homogeneous(arr_f, v_grid)
    entropy     = compute_entropy_homogeneous(arr_f, dv=dv, eps=eps)

    fig, axes = plt.subplots(2,2, figsize=figsize, dpi=dpi)
    fig.suptitle(f"Plot of several physical quantities", fontsize=suptitle_fontsize)
    
    arr_t       = arr_t.cpu()
    mass        = mass.cpu()
    momentum    = momentum.cpu()
    energy      = energy.cpu()
    entropy     = entropy.cpu()
    axes[0,0].set_title("Mass",     fontsize = title_fontsize)
    axes[0,1].set_title("Momentum", fontsize = title_fontsize)
    axes[1,0].set_title("Energy",   fontsize = title_fontsize)
    axes[1,1].set_title("Entropy",  fontsize = title_fontsize)
    if mode=="plot":
        axes[0,0].plot(arr_t, mass[:, 0],       linewidth=plot_linewidth)
        axes[0,1].plot(arr_t, momentum[:, 0],   linewidth=plot_linewidth, ls='-', c='r', label='$x$')
        axes[0,1].plot(arr_t, momentum[:, 1],   linewidth=plot_linewidth, ls='-', c='g', label='$y$')
        axes[1,0].plot(arr_t, energy[:, 0],     linewidth=plot_linewidth)
        axes[1,1].plot(arr_t, entropy[:, 0],    linewidth=plot_linewidth)
    elif mode=="scatter":
        axes[0,0].scatter(arr_t, mass[:, 0],        s=scatter_size,)
        axes[0,1].scatter(arr_t, momentum[:, 0],    s=scatter_size, c='r', label='$x$')
        axes[0,1].scatter(arr_t, momentum[:, 1],    s=scatter_size, c='g', label='$y$')
        axes[1,0].scatter(arr_t, energy[:, 0],      s=scatter_size,)
        axes[1,1].scatter(arr_t, entropy[:, 0],     s=scatter_size,)
    axes[0,1].legend()
    for i, j in product(range(2), range(2)):
        axes[i,j].set_xlabel("time index" if arr_t.dtype==torch.long else "$t$")
        axes[i,j].grid()
        axes[i,j].set_xlim(arr_t[0], arr_t[-1])
    axes[0,0].set_ylim(0, 2*mass[0,0])
    axes[0,1].set_ylim(-2*(momentum[0].norm()+0.1), +2*(momentum[0].norm()+0.1))
    axes[1,0].set_ylim(0, 2*energy[0,0])
    fig.tight_layout()

    return fig, axes


##################################################
##################################################
# End of file