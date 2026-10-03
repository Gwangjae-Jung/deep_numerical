from   typing import Optional, Sequence
import torch

from   deep_numerical import ones, repeat, zeros


EPSILON: float = 1e-20

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
def compute_moments_homogeneous(
    f:   torch.Tensor,
    v:   torch.Tensor,
    eps: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes physical quantities determining the local Maxwellian distribution: mass, velocity, and temperature.

    ## Description
    Computes density, mean velocity, and temperature moments for a spatially homogeneous distribution function.

    ## Arguments
    `f` (`torch.Tensor`): The distribution function of shape `(B, *repeat(1, dim), K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): The velocity grid of shape `(K_1, ..., K_d, d)`.
    `eps` (`float`, default: `1e-20`): Numerical epsilon to prevent zero division.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tuple `(density, velocity, temperature)`.
    """
    dim:    int          = v.shape[-1]
    dV:     float        = float(torch.prod(v[*ones(dim)] - v[*zeros(dim)]))
    v_prep: torch.Tensor = v[*repeat(None, 1 + dim), ...]
    v_axes: tuple[int, ...] = tuple(range(1 + dim, 1 + 2 * dim))

    density:     torch.Tensor = f.sum(dim=v_axes, keepdim=True) * dV
    momentum:    torch.Tensor = torch.sum(f * v_prep, dim=v_axes, keepdim=True) * dV
    velocity:    torch.Tensor = momentum / (density + eps)
    _speed_sq:   torch.Tensor = torch.sum((v_prep - velocity) ** 2, dim=-1, keepdim=True)
    temperature: torch.Tensor = torch.sum(f * _speed_sq, dim=v_axes, keepdim=True) * dV / (dim * density + eps)

    dim_squeezed: tuple[int, ...] = tuple((-(2 + k) for k in range(2 * dim)))
    return (
        density.squeeze(dim_squeezed),
        velocity.squeeze(dim_squeezed),
        temperature.squeeze(dim_squeezed),
    )


def compute_moments_inhomogeneous(
    f:   torch.Tensor,
    v:   torch.Tensor,
    eps: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes physical quantities determining the local Maxwellian distribution: mass, velocity, and temperature.

    ## Description
    Computes spatially varying density, mean velocity, and temperature moments for an inhomogeneous distribution function.

    ## Arguments
    `f` (`torch.Tensor`): The distribution function of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): The velocity grid of shape `(K_1, ..., K_d, d)`.
    `eps` (`float`, default: `1e-20`): Numerical epsilon to prevent zero division.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tuple `(density, velocity, temperature)`.
    """
    dim:    int             = v.shape[-1]
    dV:     float           = float(torch.prod(v[*ones(dim)] - v[*zeros(dim)]))
    v_prep: torch.Tensor    = v.reshape(1, *ones(dim), *v.shape)
    v_axes: tuple[int, ...] = tuple(range(1 + dim, 1 + 2 * dim))

    density:     torch.Tensor = f.sum(dim=v_axes, keepdim=True) * dV
    momentum:    torch.Tensor = torch.sum(f * v_prep, dim=v_axes, keepdim=True) * dV
    velocity:    torch.Tensor = momentum / (density + eps)
    _speed_sq:   torch.Tensor = torch.sum((v_prep - velocity) ** 2, dim=-1, keepdim=True)
    temperature: torch.Tensor = torch.sum(f * _speed_sq, dim=v_axes, keepdim=True) * dV / (dim * density + eps)

    return (
        density.squeeze(v_axes),
        velocity.squeeze(v_axes),
        temperature.squeeze(v_axes),
    )


def compute_mass_homogeneous(
    f:   torch.Tensor,
    dv:  float,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Computes mass via velocity integral of `f`.

    ## Description
    Computes the total mass of each instance by integrating `f` over the velocity domain.

    ## Arguments
    `f` (`torch.Tensor`): The distribution function of shape `(B, K_1, ..., K_d, 1)`.
    `dv` (`float`): The grid spacing in velocity space.
    `dim` (`Optional[int]`, default: `None`): Velocity dimension.

    ## Returns
    `torch.Tensor`: The mass tensor of shape `(B, 1)`.
    """
    if dim is None:
        dim = (f.ndim - 2) // 2
    dV:     float           = dv ** dim
    axes_v: tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    mass:   torch.Tensor    = torch.sum(f, dim=axes_v) * dV
    return torch.squeeze(mass, dim=axes_v)


def compute_mass_inhomogeneous(
    f:   torch.Tensor,
    dx:  float,
    dv:  float,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Computes mass via space-velocity integral.

    ## Description
    Computes the total mass by calculating the space-velocity integral of the distribution function `f`.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    `dx` (`float`): Spatial grid step size.
    `dv` (`float`): Velocity grid step size.
    `dim` (`Optional[int]`, default: `None`): Velocity dimension.

    ## Returns
    `torch.Tensor`: The mass tensor of shape `(B, 1)`.
    """
    if dim is None:
        dim = (f.ndim - 2) // 2
    dX:     float           = dx ** dim
    dV:     float           = dv ** dim
    axes_x: tuple[int, ...] = tuple((+(1 + k) for k in range(dim)))
    axes_v: tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    mass:   torch.Tensor    = torch.sum(f, dim=(*axes_x, *axes_v)) * (dX * dV)
    return mass


def compute_momentum_homogeneous(
    f: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Computes momentum via velocity integral of `f * v`.

    ## Description
    Computes total momentum of each instance by integrating $f(v) v$ over velocity space.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): Velocity coordinate grid tensor of shape `(K_1, ..., K_d, d)`.

    ## Returns
    `torch.Tensor`: The momentum tensor of shape `(B, d)`.
    """
    dim:      int             = v.shape[-1]
    dV:       torch.Tensor    = torch.prod(v[*ones(dim)] - v[*zeros(dim)])
    v_prep:   torch.Tensor    = v[None, ...]
    axes_v:   tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    momentum: torch.Tensor    = torch.sum(f * v_prep, dim=axes_v) * dV
    return torch.squeeze(momentum, dim=axes_v)


def compute_momentum_inhomogeneous(
    f:  torch.Tensor,
    v:  torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """Computes momentum via space-velocity integral of `f * v`.

    ## Description
    Computes total momentum of each instance by integrating over both spatial and velocity domains.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): Velocity coordinate grid tensor of shape `(K_1, ..., K_d, d)`.
    `dx` (`float`): Spatial step size.

    ## Returns
    `torch.Tensor`: The momentum tensor of shape `(B, d)`.
    """
    dim:      int             = v.shape[-1]
    dX:       float           = dx ** dim
    dV:       torch.Tensor    = torch.prod(v[ones(dim)] - v[zeros(dim)])
    v_prep:   torch.Tensor    = v.reshape(1, *ones(dim), *v.shape)
    axes_x:   tuple[int, ...] = tuple((+(1 + k) for k in range(dim)))
    axes_v:   tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    momentum: torch.Tensor    = torch.sum(f * v_prep, dim=(*axes_x, *axes_v), keepdim=True) * (dX * dV)
    return torch.squeeze(momentum, dim=(*axes_x, *axes_v))


def compute_energy_homogeneous(
    f: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Computes kinetic energy via velocity integral of `f * |v|^2 / 2`.

    ## Description
    Computes total kinetic energy of each instance by integrating $f(v) |v|^2 / 2$ over velocity space.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): Velocity coordinate grid tensor of shape `(K_1, ..., K_d, d)`.

    ## Returns
    `torch.Tensor`: Kinetic energy tensor of shape `(B, 1)`.
    """
    dim:      int             = v.shape[-1]
    dV:       torch.Tensor    = torch.prod(v[*ones(dim)] - v[*zeros(dim)])
    v_prep:   torch.Tensor    = v[None, ...]
    axes_v:   tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    speed_sq: torch.Tensor    = torch.sum(v_prep ** 2, dim=-1, keepdim=True)
    energy:   torch.Tensor    = torch.sum(f * speed_sq, dim=axes_v) * dV / 2
    return torch.squeeze(energy, dim=axes_v)


def compute_energy_inhomogeneous(
    f:  torch.Tensor,
    v:  torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """Computes kinetic energy via space-velocity integral of `f * |v|^2 / 2`.

    ## Description
    Computes total kinetic energy by integrating over both spatial and velocity domains.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): Velocity coordinate grid tensor of shape `(K_1, ..., K_d, d)`.
    `dx` (`float`): Spatial step size.

    ## Returns
    `torch.Tensor`: Kinetic energy tensor of shape `(B, 1)`.
    """
    dim:      int             = v.shape[-1]
    dX:       float           = dx ** dim
    dV:       torch.Tensor    = torch.prod(v[*ones(dim)] - v[*zeros(dim)])
    v_prep:   torch.Tensor    = v.reshape(1, *ones(dim), *v.shape)
    axes_x:   tuple[int, ...] = tuple((+(1 + k) for k in range(dim)))
    axes_v:   tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    speed_sq: torch.Tensor    = torch.sum(v_prep ** 2, dim=-1, keepdim=True)
    energy:   torch.Tensor    = torch.sum(f * speed_sq, dim=(*axes_x, *axes_v), keepdim=True) * (dX * dV) / 2
    return torch.squeeze(energy, dim=(*axes_x, *axes_v))


def compute_entropy_homogeneous(
    f:   torch.Tensor,
    dv:  float,
    dim: Optional[int] = None,
    eps: float         = EPSILON,
) -> torch.Tensor:
    """Computes entropy via velocity integral of `f * log(f)`.

    ## Description
    Computes Boltzmann entropy $H(f) = \\int f \\log f \\, dv$ over the velocity domain.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, *ones(d), K_1, ..., K_d, 1)`.
    `dv` (`float`): Velocity grid spacing.
    `dim` (`Optional[int]`, default: `None`): Velocity dimension.
    `eps` (`float`, default: `1e-20`): Numerical epsilon to prevent non-positive logarithm.

    ## Returns
    `torch.Tensor`: The entropy tensor of shape `(B, 1)`.
    """
    if dim is None:
        dim = (f.ndim - 2) // 2
    axes_v:  tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    dV:      float           = dv ** dim
    f_clean: torch.Tensor    = f - f.min() + eps if f.min() <= 0 else f
    entropy: torch.Tensor    = torch.sum(f_clean * f_clean.log(), dim=axes_v) * dV
    return torch.squeeze(entropy, dim=axes_v)


def compute_entropy_inhomogeneous(
    f:   torch.Tensor,
    dx:  float,
    dv:  float,
    dim: Optional[int] = None,
    eps: float         = EPSILON,
) -> torch.Tensor:
    """Computes entropy via space-velocity integral of `f * log(f)`.

    ## Description
    Computes total entropy by calculating the space-velocity integral of $f \\log f$.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(B, K_1, ..., K_d, 1)`.
    `dx` (`float`): Spatial grid step size.
    `dv` (`float`): Velocity grid step size.
    `dim` (`Optional[int]`, default: `None`): Velocity dimension.
    `eps` (`float`, default: `1e-20`): Numerical epsilon to prevent non-positive logarithm.

    ## Returns
    `torch.Tensor`: The entropy tensor of shape `(B, 1)`.
    """
    if dim is None:
        dim = (f.ndim - 2) // 2
    axes_x:  tuple[int, ...] = tuple((-(2 + k + dim) for k in range(dim)))
    axes_v:  tuple[int, ...] = tuple((-(2 + k) for k in range(dim)))
    dX:      float           = dx ** dim
    dV:      float           = dv ** dim
    f_clean: torch.Tensor    = f - f.min() + eps if f.min() <= 0 else f
    entropy: torch.Tensor    = torch.sum(f_clean * f_clean.log(), dim=(*axes_x, *axes_v), keepdim=True) * (dX * dV)
    return torch.squeeze(entropy, dim=(*axes_x, *axes_v))


def plot_quantities_homogeneous(
    arr_f:             torch.Tensor,
    v_grid:            torch.Tensor,
    arr_t:             Optional[torch.Tensor] = None,
    dim:               Optional[int]          = None,
    eps:               float                  = EPSILON,
    figsize:           tuple[int, int]        = (10, 7),
    dpi:               int                    = 100,
    mode:              str                    = 'plot',
    suptitle_fontsize: int                    = 20,
    title_fontsize:    int                    = 12,
    plot_linewidth:    float                  = 1.0,
    scatter_size:      float                  = 10.0,
) -> tuple[object, Sequence[object]]:
    """Plot physical quantities of a homogeneous distribution.

    ## Description
    Computes and plots the time evolution of physical quantities (mass, momentum components, kinetic energy, and entropy) for a spatially homogeneous distribution function.

    ## Arguments
    `arr_f` (`torch.Tensor`): The distribution tensor of shape `(batch_size, *v_grid_shape, 1)`.
    `v_grid` (`torch.Tensor`): The velocity coordinate grid tensor.
    `arr_t` (`Optional[torch.Tensor]`, default: `None`): 1D array of time steps.
    `dim` (`Optional[int]`, default: `None`): Dimension of velocity space.
    `eps` (`float`, default: `1e-20`): Numerical tolerance for entropy computation.
    `figsize` (`tuple[int, int]`, default: `(10, 7)`): Figure dimensions `(width, height)`.
    `dpi` (`int`, default: `100`): Resolution of the figure in dots per inch.
    `mode` (`str`, default: `'plot'`): Plotting mode, either `'plot'` or `'scatter'`.
    `suptitle_fontsize` (`int`, default: `20`): Font size of the main figure title.
    `title_fontsize` (`int`, default: `12`): Font size of subplot titles.
    `plot_linewidth` (`float`, default: `1.0`): Line width for `'plot'` mode.
    `scatter_size` (`float`, default: `10.0`): Marker size for `'scatter'` mode.

    ## Returns
    `tuple[object, Sequence[object]]`: Matplotlib figure and axes containing the plotted physical quantities.
    """
    mode_clean: str = mode.lower()
    assert mode_clean in ('plot', 'scatter'), f"Invalid mode: {mode_clean}. Choose 'plot' or 'scatter'."
    from itertools import product
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Package 'matplotlib' is required for 'plot_quantities_homogeneous'. Please install it.")

    if dim is None:
        dim = (arr_f.ndim - 2) // 2
    if arr_t is None:
        arr_t_tensor: torch.Tensor = torch.arange(arr_f.shape[0], dtype=torch.long, device=arr_f.device)
    else:
        arr_t_tensor = arr_t

    dv:       float        = float((v_grid[*ones(dim)] - v_grid[*zeros(dim)])[0].item())
    mass:     torch.Tensor = compute_mass_homogeneous(arr_f, dv=dv, dim=dim)
    momentum: torch.Tensor = compute_momentum_homogeneous(arr_f, v_grid)
    energy:   torch.Tensor = compute_energy_homogeneous(arr_f, v_grid)
    entropy:  torch.Tensor = compute_entropy_homogeneous(arr_f, dv=dv, eps=eps)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Plot of several physical quantities", fontsize=suptitle_fontsize)

    arr_t_cpu:    torch.Tensor = arr_t_tensor.cpu()
    mass_cpu:     torch.Tensor = mass.cpu()
    momentum_cpu: torch.Tensor = momentum.cpu()
    energy_cpu:   torch.Tensor = energy.cpu()
    entropy_cpu:  torch.Tensor = entropy.cpu()

    axes[0, 0].set_title("Mass",     fontsize=title_fontsize)
    axes[0, 1].set_title("Momentum", fontsize=title_fontsize)
    axes[1, 0].set_title("Energy",   fontsize=title_fontsize)
    axes[1, 1].set_title("Entropy",  fontsize=title_fontsize)

    if mode_clean == "plot":
        axes[0, 0].plot(arr_t_cpu, mass_cpu[:, 0],     linewidth=plot_linewidth)
        axes[0, 1].plot(arr_t_cpu, momentum_cpu[:, 0], linewidth=plot_linewidth, ls='-', c='r', label='$x$')
        axes[0, 1].plot(arr_t_cpu, momentum_cpu[:, 1], linewidth=plot_linewidth, ls='-', c='g', label='$y$')
        axes[1, 0].plot(arr_t_cpu, energy_cpu[:, 0],   linewidth=plot_linewidth)
        axes[1, 1].plot(arr_t_cpu, entropy_cpu[:, 0],  linewidth=plot_linewidth)
    elif mode_clean == "scatter":
        axes[0, 0].scatter(arr_t_cpu, mass_cpu[:, 0],     s=scatter_size)
        axes[0, 1].scatter(arr_t_cpu, momentum_cpu[:, 0], s=scatter_size, c='r', label='$x$')
        axes[0, 1].scatter(arr_t_cpu, momentum_cpu[:, 1], s=scatter_size, c='g', label='$y$')
        axes[1, 0].scatter(arr_t_cpu, energy_cpu[:, 0],   s=scatter_size)
        axes[1, 1].scatter(arr_t_cpu, entropy_cpu[:, 0],  s=scatter_size)

    axes[0, 1].legend()
    for i, j in product(range(2), range(2)):
        axes[i, j].set_xlabel("time index" if arr_t_cpu.dtype == torch.long else "$t$")
        axes[i, j].grid()
        axes[i, j].set_xlim(arr_t_cpu[0], arr_t_cpu[-1])
    axes[0, 0].set_ylim(0, 2 * mass_cpu[0, 0])
    axes[0, 1].set_ylim(-2 * (momentum_cpu[0].norm() + 0.1), +2 * (momentum_cpu[0].norm() + 0.1))
    axes[1, 0].set_ylim(0, 2 * energy_cpu[0, 0])
    fig.tight_layout()

    return fig, axes


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()