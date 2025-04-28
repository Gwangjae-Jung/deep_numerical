import  numpy           as np
import  torch


__all__: list[str] = ['sample_quantities', 'sample_noise_quadratic', 'compute_quadratic_polynomial', 'normalize_density']


def sample_quantities(
        dimension:      int,
        batch_size:     int,
        v_perturb_max:  float = 1.0,
        t_perturb_max:  float = 0.2,
        
        dtype:      torch.dtype     = torch.double,
        device:     torch.device    = torch.device('cpu'),
    ) -> tuple[torch.Tensor]:
    """Sample the mean density, velocity, and temperatures."""
    dtype_and_device = {'dtype': dtype, 'device': device}
    rho = torch.ones((batch_size, 1), **dtype_and_device)
    u   = v_perturb_max *(2*torch.rand(batch_size, dimension, **dtype_and_device) - 1)
    t   = (1 + (t_perturb_max**0.5) * (2*torch.rand(batch_size, 1, **dtype_and_device) - 1))**2
    return rho, u, t


def sample_noise_quadratic(
        dimension:      int,
        v_max:          float,
        batch_size:     int,
        
        dtype:      torch.dtype     = torch.double,
        device:     torch.device    = torch.device('cpu'),
    ) -> tuple[torch.Tensor]:
    """Sample the quadratic noise $q$, for which $1 + q >= 0$."""
    dim_simplex = int(np.sum([np.power(dimension, i) for i in range(0, dimension+1)]))
    sampler = torch.distributions.Dirichlet(torch.ones(dim_simplex+1))
    
    __coeffs = sampler.sample((batch_size,))[:, :-1]
    __coeffs = __coeffs.type(dtype).to(device)
    __rand_sign = 2*torch.randint(0, 2, size=__coeffs.shape, device=device)-1
    __coeffs = __coeffs * __rand_sign
    
    __split_arg = tuple((dimension**k for k in range(0, 2+1)))
    coeffs = torch.split(__coeffs, __split_arg, dim=-1)
    
    return tuple((
        torch.reshape(
            c/(v_max**k),
            (batch_size, *(dimension for _ in range(k)))
        )
        for k, c in enumerate(coeffs)
    ))


def compute_quadratic_polynomial(x: torch.Tensor, coeffs: tuple[torch.Tensor]) -> torch.Tensor:
    """Compute the quadratic polynomial $q(x) = c_0 + c_1 x + c_2 x^2$.
    """
    c0 = coeffs[0]
    c1 = coeffs[1]
    c2 = coeffs[2]
    assert c0.size(0)==c1.size(0) and c1.size(0)==c2.size(0)
    dimension = x.size(-1)
    ord0 = c0.reshape(-1, *(1 for _ in range(dimension)))
    ord1 = torch.einsum("...i, bi -> b...", x, c1)
    ord2 = torch.einsum("...i, bij, ...j -> b...", x, c2, x)
    return ord0+ord1+ord2


def normalize_density(f: torch.Tensor, delta_v: float) -> torch.Tensor:
    """Normalize the density to 1.
    """
    dimension = (f.ndim-2)//2
    dv: float = delta_v**dimension
    density = f.sum(tuple(range(-1-dimension, -1))) * dv
    density = density.reshape(-1, *(1 for _ in range(2*dimension+1)))
    return f/density


##################################################
##################################################
# End of file