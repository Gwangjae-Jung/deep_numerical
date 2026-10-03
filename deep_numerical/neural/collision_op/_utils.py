import torch

from   deep_numerical import ones, zeros


__all__: list[str] = ["compute_moments_homogeneous", "maxwellian_homogeneous"]



##################################################
def compute_moments_homogeneous(
        f:      torch.Tensor,
        v:      torch.Tensor,
        eps:    float = 1e-20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes hydrodynamic moments determining local Maxwellian: density, velocity, and temperature.

    ## Description
    Computes mass density `rho`, mean flow velocity `u`, and kinetic temperature `T` for homogeneous distribution functions.

    ## Arguments
    `f` (`torch.Tensor`): Distribution function tensor of shape `(B, K_1, ..., K_d, 1)`.
    `v` (`torch.Tensor`): Velocity coordinate grid tensor of shape `(K_1, ..., K_d, d)`.
    `eps` (`float`, default: `1e-20`): Regularization epsilon to avoid division by zero.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tuple `(density, velocity, temperature)` where shapes are `(B, 1)`, `(B, d)`, and `(B, 1)`.
    """    
    # Retrieve the dimension and `dV`
    dim = v.shape[-1]
    dV  = float(torch.prod(v[*ones(dim)] - v[*zeros(dim)]))
    
    # Reshape `v` in this function for vectorized operations\
    v_axes = tuple(range(1, 1+dim)) # The following `dim` dimensions
    
    # Compute the density
    density: torch.Tensor = f.sum(axis=v_axes, keepdims=True) * dV
    
    # Compute the average velocity
    momentum: torch.Tensor = torch.sum(f*v, axis=v_axes, keepdims=True) * dV
    velocity: torch.Tensor = momentum / (density + eps)
    
    # Compute the temperature
    _speed_sq:   torch.Tensor = torch.sum((v-velocity)**2, axis=-1, keepdims=True)
    temperature: torch.Tensor = torch.sum(f*_speed_sq, axis=v_axes, keepdims=True) * dV / (dim*density + eps)
    
    # Squeeze the dimensions (`...{*v}c->...c`) and return
    dim_squeezed = tuple((-(2+k) for k in range(dim)))
    density     = density.squeeze(dim_squeezed)
    velocity    = velocity.squeeze(dim_squeezed)
    temperature = temperature.squeeze(dim_squeezed)
    return (density, velocity, temperature)


def maxwellian_homogeneous(
        v:                  torch.Tensor,
        mean_density:       torch.Tensor,
        mean_velocity:      torch.Tensor,
        mean_temperature:   torch.Tensor,
        eps:                float   = 1e-20,
    ) -> torch.Tensor:
    """Computes the local Maxwellian equilibrium distribution for homogeneous arguments.

    ## Description
    Constructs the discretized Maxwellian equilibrium state from given mean density, velocity, and temperature.

    ## Arguments
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(K_1, ..., K_d, d)`.
    `mean_density` (`torch.Tensor`): Mean density tensor of shape `(B, 1)`.
    `mean_velocity` (`torch.Tensor`): Mean velocity tensor of shape `(B, d)`.
    `mean_temperature` (`torch.Tensor`): Mean temperature tensor of shape `(B, 1)`.
    `eps` (`float`, default: `1e-20`): Regularization epsilon to avoid division by zero.

    ## Returns
    `torch.Tensor`: Maxwellian distribution tensor of shape `(B, K_1, ..., K_d, 1)`.
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
        mean_density.reshape(    num_instances, *ones(dim), 1)
    mean_velocity:      torch.Tensor  = \
        mean_velocity.reshape(   num_instances, *ones(dim), dim)
    mean_temperature:   torch.Tensor  = \
        mean_temperature.reshape(num_instances, *ones(dim), 1)
    
    # Compute the Maxwellian
    _scale:     torch.Tensor  = \
        mean_density / torch.pow(2*torch.pi*mean_temperature+eps, dim/2)
    _exp:       torch.Tensor  = \
        -torch.sum(
            (v[None, ...]-mean_velocity)**2,
            dim=-1, keepdims=True
        ) / (2*mean_temperature)
    ret = _scale * torch.exp(_exp)
    
    # Return the result
    return ret


##################################################
def main() -> None:
    from deep_numerical.utils import space_grid
    batch_size: int   = 3
    v_max:      float = 6.0
    rho: torch.Tensor = torch.randn((batch_size, 1)) * 1e-2 + 1.0
    vel: torch.Tensor = torch.randn((batch_size, 2)) * 1e-2 + torch.randn((batch_size, 2))
    temp: torch.Tensor = torch.randn((batch_size, 1)) * 1e-2 + 1.0

    v_grid: torch.Tensor = space_grid(2, 128, v_max)
    dists:  torch.Tensor = maxwellian_homogeneous(v_grid, rho, vel, temp)
    print(f"Computed Maxwellian dists with shape: {dists.shape}")


if __name__ == '__main__':
    main()