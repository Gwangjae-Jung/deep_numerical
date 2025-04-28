import  numpy           as np


__all__: list[str] = ['sample_quantities', 'sample_noise_quadratic', 'compute_quadratic_polynomial', 'normalize_density']


def sample_quantities(
        dimension:      int,
        batch_size:     int,
        v_perturb_max:  float = 1.0,
        t_perturb_max:  float = 0.2,
    ) -> tuple[np.ndarray]:
    """Sample the mean density, velocity, and temperatures."""
    rho = np.ones((batch_size, 1))
    u   = v_perturb_max *(2*np.random.rand(batch_size, dimension) - 1)
    t   = (1 + (t_perturb_max**0.5) * (2*np.random.rand(batch_size, 1) - 1))**2
    return rho, u, t


def sample_noise_quadratic(
        dimension:      int,
        v_max:          float,
        batch_size:     int,
    ) -> tuple[np.ndarray]:
    """Sample the quadratic noise $q$, for which $1 + q >= 0$."""
    dim_simplex = int(np.sum([np.power(dimension, i) for i in range(0, dimension+1)]))
    __coeffs = np.random.dirichlet(np.ones(dim_simplex+1), size=batch_size)
    __rand_sign = 2*np.random.randint(0, 2, size=__coeffs.shape)-1
    __coeffs = __coeffs * __rand_sign
    
    __split_arg = tuple((dimension**k for k in range(0, 2+1)))
    coeffs = np.split(__coeffs, np.cumsum(__split_arg), axis=-1)
    coeffs.pop(-1)
    
    for c in coeffs:
        print(c.shape)
    
    return tuple((
        np.reshape(
            c/(v_max**k),
            (batch_size, *(dimension for _ in range(k)))
        )
        for k, c in enumerate(coeffs)
    ))


def compute_quadratic_polynomial(x: np.ndarray, coeffs: tuple[np.ndarray]) -> np.ndarray:
    """Compute the quadratic polynomial $q(x) = c_0 + c_1 x + c_2 x^2$.
    """
    c0 = coeffs[0]
    c1 = coeffs[1]
    c2 = coeffs[2]
    assert c0.shape[0]==c1.shape[0] and c1.shape[0]==c2.shape[0]
    dimension = x.shape[-1]
    ord0 = c0.reshape(-1, *(1 for _ in range(dimension)))
    ord1 = np.einsum("...i, bi -> b...", x, c1)
    ord2 = np.einsum("...i, bij, ...j -> b...", x, c2, x)
    return ord0+ord1+ord2


def normalize_density(f: np.ndarray, delta_v: float) -> np.ndarray:
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