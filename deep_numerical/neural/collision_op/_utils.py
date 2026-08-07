import  torch
from    deep_numerical  import  zeros, ones


__all__ = ["compute_moments_homogeneous", "maxwellian_homogeneous"]


##################################################
##################################################
def compute_moments_homogeneous(
        f:      torch.Tensor,
        v:      torch.Tensor,
        eps:    float = 1e-20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the physical quantities which determine the local Maxwellian distribution - mass, velocity, and temperature.
    
    Arguments:
        `f` (`torch.Tensor`):
            The distribution function at a specific time, which is of shape `(B, K_1, ..., K_d, 1)`.
        `v` (`torch.Tensor`):
            The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `eps` (`float`, default: `1e-20`):
            The value which is used to prevent divisions by zero.
    
    Returns:
        This function returns the tuple `(rho, u, T)`, where `rho`, `u`, and `T` are `torch.Tensor` objects described below.
            * `rho` saves the mean density of each instance, which is of shape `(B, 1)`.
            * `u` saves the mean velocity of each instance, which is of shape `(B, d)`.
            * `T` saves the mean temperature of each instance, which is of shape `(B, 1)`.
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
    """Compute the local Maxwellian distribution with homogeneous input arguments.
    
    -----
    ### Note
    In this implementation, the tensors `mean_density`, `mean_velocity`, and `mean_temperature` should of shape `(B, x)` where `x==dim` for `mean_velocity` and `x==1` for the others.
    
    Arguments:
        `v` (`torch.Tensor`): The velocity grid, which is of shape `(K_1, ..., K_d, d)`.
        `mean_density` (`torch.Tensor`): The mean density, which is of shape `(B, 1)`.
        `mean_velocity` (`torch.Tensor`): The mean velocity, which is of shape `(B, d)`.
        `mean_temperature` (`torch.Tensor`): The mean temperature, which is of shape `(B, 1)`.
        `eps` (`float`, default: `1e-20`): The value which is used to prevent divisions by zero.
    
    Returns:
        `torch.Tensor`: This function returns a `torch.Tensor` object of shape `(1, K_1, ..., K_d, 1)`, which is the discretization of the BKW solution corresponding to the input arguments.
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
##################################################
if __name__=="__main__":
    from    deep_numerical.utils    import  space_grid
    import  matplotlib.pyplot       as      plt
    batch_size = 3
    v_max = 6.0
    rho = torch.randn((batch_size, 1))*1e-2 + 1.0
    u = torch.randn((batch_size, 2))*1e-2 + torch.randn((batch_size, 2))
    T = torch.randn((batch_size, 1))*1e-2 + 1.0
    
    v = space_grid(2, 128, v_max)
    dists = maxwellian_homogeneous(v, rho, u, T)
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    ax: plt.Axes
    for ax, dist in zip(axes, dists):
        ax.imshow(dist[..., 0], extent=(-v_max, v_max, -v_max, v_max), origin='lower')
        ax.set_xticks([-v_max, 0, v_max], [r"$-v_\text{max}$", "0", r"$v_\text{max}$"])
        ax.set_yticks([-v_max, 0, v_max], [r"$-v_\text{max}$", "0", r"$v_\text{max}$"])
    fig.tight_layout()
    fig.savefig("maxwellian_test.png", dpi=1000)
    print("Saved the Maxwellian distribution test figure as 'maxwellian_test.png'.")
    
    
##################################################
##################################################
# End of file