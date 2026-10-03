from   typing import Literal
import torch

from   deep_numerical import ones, repeat


__all__: list[str] = [
    'maxwellian_homogeneous',
    'maxwellian_inhomogeneous',
    'bkw',
    'get_bkw_coeff_int',
    'get_bkw_coeff_ext',
]


##################################################
def maxwellian_homogeneous(
    v:                torch.Tensor,
    mean_density:     torch.Tensor,
    mean_velocity:    torch.Tensor,
    mean_temperature: torch.Tensor,
) -> torch.Tensor:
    """Compute the local Maxwellian distribution with homogeneous input arguments.

    ## Description
    Computes the Maxwellian distribution over velocity grid `v` given instance-wise macroscopic quantities.

    ## Arguments
    `v` (`torch.Tensor`): The velocity grid of shape `(K_1, ..., K_d, d)`.
    `mean_density` (`torch.Tensor`): Mean density of shape `(B, 1)`.
    `mean_velocity` (`torch.Tensor`): Mean velocity of shape `(B, d)`.
    `mean_temperature` (`torch.Tensor`): Mean temperature of shape `(B, 1)`.

    ## Returns
    `torch.Tensor`: Maxwellian distribution tensor of shape `(B, *ones(2*dim), 1)`.
    """
    if not (
        mean_density.shape[0] == mean_velocity.shape[0]
        and mean_velocity.shape[0] == mean_temperature.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: mean_density={mean_density.shape}, mean_velocity={mean_velocity.shape}, mean_temperature={mean_temperature.shape}"
        )

    num_instances: int = mean_density.shape[0]
    dim:           int = v.shape[-1]

    if not (mean_density.ndim == 2 and mean_density.shape[-1] == 1):
        raise ValueError(f"'mean_density' should be a 2-dimensional array of shape (B, 1), but got {mean_density.shape}.")
    if not (mean_velocity.ndim == 2 and mean_velocity.shape[-1] == dim):
        raise ValueError(f"The mean velocity should be a 2-dimensional array of shape (B, dim), but got {mean_velocity.shape}.")
    if not (mean_temperature.ndim == 2 and mean_temperature.shape[-1] == 1):
        raise ValueError(f"The mean temperature should be a 2-dimensional array of shape (B, 1), but got {mean_temperature.shape}.")

    mean_density_reshaped:     torch.Tensor = mean_density.reshape(num_instances, *ones(2 * dim), 1)
    mean_velocity_reshaped:    torch.Tensor = mean_velocity.reshape(num_instances, *ones(2 * dim), dim)
    mean_temperature_reshaped: torch.Tensor = mean_temperature.reshape(num_instances, *ones(2 * dim), 1)

    _scale: torch.Tensor = mean_density_reshaped / torch.pow(2 * torch.pi * mean_temperature_reshaped, dim / 2)
    _exp:   torch.Tensor = -torch.sum(
        (v[*repeat(None, 1 + dim), ...] - mean_velocity_reshaped) ** 2,
        dim     = -1,
        keepdim = True,
    ) / (2 * mean_temperature_reshaped)
    return _scale * torch.exp(_exp)


def maxwellian_inhomogeneous(
    xv:               torch.Tensor,
    mean_density:     torch.Tensor,
    mean_velocity:    torch.Tensor,
    mean_temperature: torch.Tensor,
    eps:              float = 1e-20,
) -> torch.Tensor:
    """Compute the local Maxwellian distribution with inhomogeneous input arguments.

    ## Description
    Computes the local Maxwellian distribution over the spatio-velocity grid `xv` from spatially varying macroscopic fields.

    ## Arguments
    `xv` (`torch.Tensor`): Spatio-velocity grid of shape `(N_1, ..., N_d, K_1, ..., K_d, 2*d)`.
    `mean_density` (`torch.Tensor`): Local mean density of shape `(B, N_1, ..., N_d, 1)`.
    `mean_velocity` (`torch.Tensor`): Local mean velocity of shape `(B, N_1, ..., N_d, d)`.
    `mean_temperature` (`torch.Tensor`): Local mean temperature of shape `(B, N_1, ..., N_d, 1)`.
    `eps` (`float`, default: `1e-20`): Numerical stability epsilon.

    ## Returns
    `torch.Tensor`: Local Maxwellian tensor of shape `(B, N_1, ..., N_d, K_1, ..., K_d, 1)`.
    """
    if not (
        mean_density.shape[0] == mean_velocity.shape[0]
        and mean_velocity.shape[0] == mean_temperature.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: mean_density={mean_density.shape}, mean_velocity={mean_velocity.shape}, mean_temperature={mean_temperature.shape}"
        )

    num_instances: int             = mean_density.shape[0]
    dim:           int             = xv.shape[-1] // 2
    space_res:     tuple[int, ...] = xv.shape[:dim]

    if not (mean_density.ndim == dim + 2 and mean_density.shape[-1] == 1):
        raise ValueError(f"'mean_density' should be of shape (B, N_1, ..., N_d, 1), but got {mean_density.shape}.")
    if not (mean_velocity.ndim == dim + 2 and mean_velocity.shape[-1] == dim):
        raise ValueError(f"'mean_velocity' should be of shape (B, N_1, ..., N_d, dim), but got {mean_velocity.shape}.")
    if not (mean_temperature.ndim == dim + 2 and mean_temperature.shape[-1] == 1):
        raise ValueError(f"'mean_temperature' should be of shape (B, N_1, ..., N_d, 1), but got {mean_temperature.shape}.")

    mean_density_reshaped:     torch.Tensor = mean_density.reshape(num_instances, *space_res, *ones(dim), 1)
    mean_velocity_reshaped:    torch.Tensor = mean_velocity.reshape(num_instances, *space_res, *ones(dim), dim)
    mean_temperature_reshaped: torch.Tensor = mean_temperature.reshape(num_instances, *space_res, *ones(dim), 1)

    _scale: torch.Tensor = mean_density_reshaped / ((2 * torch.pi * mean_temperature_reshaped) ** (dim / 2) + eps)
    _exp:   torch.Tensor = -torch.sum((xv[..., dim:] - mean_velocity_reshaped) ** 2, dim=-1, keepdim=True) / (2 * mean_temperature_reshaped + eps)
    return _scale * torch.exp(_exp)


def bkw(
    t:         torch.Tensor,
    v:         torch.Tensor,
    kernel:    float,
    coeff_ext: float,
    density:   float                                              = 1.0,
    verbose:   bool                                               = True,
    equation:  Literal['boltzmann', 'fpl', 'fokker-planck-landau'] = 'boltzmann',
) -> torch.Tensor:
    """Returns an array of values of the BKW solution.

    ## Description
    Evaluates the analytical Bobylev-Krook-Wu (BKW) exact solution for the homogeneous kinetic equations for Maxwellian gas.

    ## Arguments
    `t` (`torch.Tensor`): Time grid tensor of shape `(num_timesteps,)`.
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(K_1, ..., K_d, d)`.
    `kernel` (`float`): Collision kernel scale.
    `coeff_ext` (`float`): Outer exponential coefficient.
    `density` (`float`, default: `1.0`): Macroscopic density.
    `verbose` (`bool`, default: `True`): Whether to display coefficients.
    `equation` (`Literal['boltzmann', 'fpl', 'fokker-planck-landau']`, default: `'boltzmann'`): Kinetic equation model.

    ## Returns
    `torch.Tensor`: BKW solution tensor of shape `(1, num_timesteps, *ones(d), K_1, ..., K_d, 1)`.
    """
    equation_str: str   = equation.lower()
    dim:          int   = v.shape[-1]
    coeff_int:    float = get_bkw_coeff_int(dim, kernel, equation_str, density)

    if verbose:
        line:  str = '-' * 10
        front: str = ''.join((line, "[ BKW solution ]", line))
        back:  str = '-' * len(front)
        print(front)
        print(f"* coeff_ext: {coeff_ext}")
        print(f"* coeff_int: {coeff_int}")
        print(back)

    _t:        torch.Tensor = t.reshape(1, -1, *ones(dim), *ones(dim), 1)
    _speed_sq: torch.Tensor = torch.reshape(torch.sum(v ** 2, dim=-1), (1, 1, *repeat(1, dim), *v.shape[:-1], 1))
    k_t:       torch.Tensor = 1 - coeff_ext * torch.exp(-coeff_int * _t)

    _part1: torch.Tensor = density * torch.pow(2 * torch.pi * k_t, -dim / 2)
    _part2: torch.Tensor = torch.exp(-_speed_sq / (2 * k_t))
    _part3: torch.Tensor = ((dim + 2) / 2) - ((dim + _speed_sq) / 2) / k_t + (_speed_sq / 2) / (k_t ** 2)

    return _part1 * _part2 * _part3


def get_bkw_coeff_int(
    dim:      int,
    kernel:   float,
    equation: Literal['boltzmann', 'fpl', 'fokker-planck-landau'],
    density:  float = 1.0,
) -> float:
    """Computes the interior exponential coefficient in the BKW solution.

    ## Description
    Calculates the time decay coefficient in the exponential term of the BKW analytical solution.

    ## Arguments
    `dim` (`int`): Dimension of the velocity domain.
    `kernel` (`float`): Collision kernel value.
    `equation` (`Literal['boltzmann', 'fpl', 'fokker-planck-landau']`): Kinetic equation model.
    `density` (`float`, default: `1.0`): Macroscopic density.

    ## Returns
    `float`: Time decay coefficient in exponent.
    """
    coeff_int: float
    if equation in ['boltzmann']:
        if dim == 2:
            coeff_int = float(torch.pi / 4) * kernel
        elif dim == 3:
            coeff_int = float(2 * torch.pi / 3) * kernel
        else:
            raise NotImplementedError(f"This function is implemented only for 2D or 3D Maxwellian gas: dim={dim}")
    elif equation in ['fpl', 'fokker-planck-landau']:
        coeff_int = float(2 * (dim - 1) * kernel)
    else:
        raise ValueError(f"Unsupported equation: {equation}")
    coeff_int *= density
    return coeff_int


def get_bkw_coeff_ext(dim: int) -> float:
    """Get BKW relaxation coefficient.

    ## Description
    Returns the exact relaxation coefficient for the BKW solution in 2D or 3D.

    ## Arguments
    `dim` (`int`): Dimension of the domain (2 or 3).

    ## Returns
    `float`: The relaxation coefficient for the BKW solution.
    """
    if dim == 2:
        return 0.5
    elif dim == 3:
        return 1.0
    else:
        raise NotImplementedError(f"Check the dimension: dim={dim}")


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()