from   typing                                    import Callable, Optional
import torch
from   torch.special                             import bessel_j0 as j0

from   deep_numerical                            import ones
from   deep_numerical.fft                        import freq_index_pair_tensor, freq_pair_tensor
from   deep_numerical.numerical.integrate        import integration_guass_legendre
from   deep_numerical.numerical.math             import sinc
from   deep_numerical.numerical.solver.constants import LAMBDA


__all__: list[str] = [
    'Boltzmann_VHS_kernel_modes_1D_integrand',
    'Boltzmann_VHS_kernel_modes_2D_integrand',
    'Boltzmann_VHS_kernel_modes_3D_integrand',
    'Boltzmann_VHS_kernel_modes',
]


##################################################
def Boltzmann_VHS_kernel_modes_1D_integrand(
    r:             torch.Tensor,
    num_grid:      int,
    v_max:         float,
    vhs_coeff:     float,
    vhs_alpha:     float,
    diagonal_only: bool                   = False,
    dtype:         Optional[torch.dtype]  = None,
    device:        Optional[torch.device] = None,
) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 1.

    ## Description
    Computes the 1D integrand value for the numerical integration of the VHS kernel modes.

    ## Arguments
    `r` (`torch.Tensor`): Integration radius variable.
    `num_grid` (`int`): Grid resolution in velocity space.
    `v_max` (`float`): Truncation boundary for velocity.
    `vhs_coeff` (`float`): VHS model coefficient $C_\\gamma$.
    `vhs_alpha` (`float`): VHS model exponent $\\alpha$.
    `diagonal_only` (`bool`, default: `False`): Whether to evaluate only self-frequency pairs.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Computation data type.
    `device` (`Optional[torch.device]`, default: `None`): Computation device.

    ## Returns
    `torch.Tensor`: Evaluated integrand tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    dimension: int = 1
    if diagonal_only:
        _r = r.reshape(*ones(dimension), -1)
    else:
        _r = r.reshape(*ones(2 * dimension), -1)
    _prod1 = (
        vhs_coeff
        * ((2 * LAMBDA * v_max) ** (dimension + vhs_alpha))
        * torch.pow(_r, (dimension - 1) + vhs_alpha)
    )

    freq_pair = freq_pair_tensor(
        dimension     = dimension,
        num_grid      = num_grid,
        keepdim       = True,
        diagonal_only = diagonal_only,
        dtype         = dtype,
        device        = device,
    )
    freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
    r_freq1 = (LAMBDA * torch.pi) * torch.norm(
        freq_pair_1 + freq_pair_2,
        p = 2, dim = -1, keepdim = True,
    )
    r_freq2 = (LAMBDA * torch.pi) * torch.norm(
        freq_pair_1 - freq_pair_2,
        p = 2, dim = -1, keepdim = True,
    )
    del freq_pair_1, freq_pair_2, freq_pair

    _prod2a = 2 * torch.cos(r_freq1 * _r)
    _prod2b = 2 * torch.cos(r_freq2 * _r)
    _prod2  = _prod2a * _prod2b

    return _prod1 * _prod2


def Boltzmann_VHS_kernel_modes_2D_integrand(
    r:             torch.Tensor,
    num_grid:      int,
    v_max:         float,
    vhs_coeff:     float,
    vhs_alpha:     float,
    diagonal_only: bool                   = False,
    dtype:         Optional[torch.dtype]  = None,
    device:        Optional[torch.device] = None,
) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 2.

    ## Description
    Computes the 2D integrand value for the numerical integration of the VHS kernel modes.

    ## Arguments
    `r` (`torch.Tensor`): Integration radius variable.
    `num_grid` (`int`): Grid resolution in velocity space.
    `v_max` (`float`): Truncation boundary for velocity.
    `vhs_coeff` (`float`): VHS model coefficient $C_\\gamma$.
    `vhs_alpha` (`float`): VHS model exponent $\\alpha$.
    `diagonal_only` (`bool`, default: `False`): Whether to evaluate only self-frequency pairs.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Computation data type.
    `device` (`Optional[torch.device]`, default: `None`): Computation device.

    ## Returns
    `torch.Tensor`: Evaluated integrand tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    dimension: int = 2
    if diagonal_only:
        _r = r.reshape(*ones(dimension), -1)
    else:
        _r = r.reshape(*ones(2 * dimension), -1)
    _prod1 = (
        vhs_coeff
        * ((2 * LAMBDA * v_max) ** (dimension + vhs_alpha))
        * torch.pow(_r, (dimension - 1) + vhs_alpha)
    )

    freq_pair = freq_pair_tensor(
        dimension     = dimension,
        num_grid      = num_grid,
        keepdim       = True,
        diagonal_only = diagonal_only,
        dtype         = dtype,
        device        = device,
    )
    freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
    r_freq1 = (LAMBDA * torch.pi) * torch.norm(
        freq_pair_1 + freq_pair_2,
        p = 2, dim = -1, keepdim = True,
    )
    r_freq2 = (LAMBDA * torch.pi) * torch.norm(
        freq_pair_1 - freq_pair_2,
        p = 2, dim = -1, keepdim = True,
    )
    del freq_pair_1, freq_pair_2, freq_pair

    _prod2a = (2 * torch.pi) * j0(r_freq1 * _r)
    _prod2b = (2 * torch.pi) * j0(r_freq2 * _r)
    _prod2  = _prod2a * _prod2b

    return _prod1 * _prod2


def Boltzmann_VHS_kernel_modes_3D_integrand(
    r:             torch.Tensor,
    num_grid:      int,
    v_max:         float,
    vhs_coeff:     float,
    vhs_alpha:     float,
    diagonal_only: bool                   = False,
    dtype:         Optional[torch.dtype]  = None,
    device:        Optional[torch.device] = None,
) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 3.

    ## Description
    Computes the 3D integrand value for the numerical integration of the VHS kernel modes.

    ## Arguments
    `r` (`torch.Tensor`): Integration radius variable.
    `num_grid` (`int`): Grid resolution in velocity space.
    `v_max` (`float`): Truncation boundary for velocity.
    `vhs_coeff` (`float`): VHS model coefficient $C_\\gamma$.
    `vhs_alpha` (`float`): VHS model exponent $\\alpha$.
    `diagonal_only` (`bool`, default: `False`): Whether to evaluate only self-frequency pairs.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Computation data type.
    `device` (`Optional[torch.device]`, default: `None`): Computation device.

    ## Returns
    `torch.Tensor`: Evaluated integrand tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    dimension: int = 3
    if diagonal_only:
        _r = r.reshape(*ones(dimension), -1)
    else:
        _r = r.reshape(*ones(2), -1)
    _prod1 = (
        vhs_coeff
        * ((2 * LAMBDA * v_max) ** (dimension + vhs_alpha))
        * torch.pow(_r, (dimension - 1) + vhs_alpha)
    )

    if diagonal_only:
        freq_pair = freq_pair_tensor(
            dimension     = dimension,
            num_grid      = num_grid,
            keepdim       = True,
            diagonal_only = True,
            dtype         = dtype,
            device        = device,
        )
        freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
        r_freq1 = (LAMBDA * torch.pi) * torch.norm(
            freq_pair_1 + freq_pair_2,
            p = 2, dim = -1, keepdim = True,
        )
        r_freq2 = (LAMBDA * torch.pi) * torch.norm(
            freq_pair_1 - freq_pair_2,
            p = 2, dim = -1, keepdim = True,
        )
        del freq_pair_1, freq_pair_2, freq_pair
        _prod2a = (4 * torch.pi) * sinc(r_freq1 * _r)
        _prod2b = (4 * torch.pi) * sinc(r_freq2 * _r)
        _prod2  = _prod2a * _prod2b
    else:
        idx_pair = freq_index_pair_tensor(
            dimension     = dimension,
            num_grid      = num_grid,
            diagonal_only = False,
        ).to(torch.float64)
        r_freq1 = (LAMBDA * torch.pi) * torch.sqrt(idx_pair[..., [0]])
        r_freq2 = (LAMBDA * torch.pi) * torch.sqrt(idx_pair[..., [1]])
        del idx_pair
        _prod2a = (4 * torch.pi) * sinc(r_freq1 * _r)
        _prod2b = (4 * torch.pi) * sinc(r_freq2 * _r)
        _prod2  = _prod2a * _prod2b

    return _prod1 * _prod2


def Boltzmann_VHS_kernel_modes(
    dimension:     int,
    num_grid:      int,
    v_max:         float,
    vhs_coeff:     float,
    vhs_alpha:     float,
    num_roots:     Optional[int]          = None,
    diagonal_only: bool                   = False,
    dtype:         Optional[torch.dtype]  = None,
    device:        Optional[torch.device] = None,
) -> torch.Tensor:
    """Computes the kernel modes for a given set of configurations.

    ## Description
    Integrates the kernel modes for the VHS collision model across the velocity domain using Gauss-Legendre quadrature.

    ## Arguments
    `dimension` (`int`): Dimension of the domain (1, 2, or 3).
    `num_grid` (`int`): Grid resolution along each velocity dimension.
    `v_max` (`float`): Truncation boundary for velocity.
    `vhs_coeff` (`float`): VHS model coefficient $C_\\gamma$.
    `vhs_alpha` (`float`): VHS model exponent $\\alpha$.
    `num_roots` (`Optional[int]`, default: `None`): Number of quadrature roots.
    `diagonal_only` (`bool`, default: `False`): Whether to evaluate only self-frequency pairs.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Computation data type.
    `device` (`Optional[torch.device]`, default: `None`): Computation device.

    ## Returns
    `torch.Tensor`: The computed kernel mode tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    if num_roots is None:
        num_roots = num_grid

    func: Callable[..., torch.Tensor]
    if dimension == 1:
        func = Boltzmann_VHS_kernel_modes_1D_integrand
    elif dimension == 2:
        func = Boltzmann_VHS_kernel_modes_2D_integrand
    elif dimension == 3:
        func = Boltzmann_VHS_kernel_modes_3D_integrand
    else:
        raise ValueError(
            f"The computation of the kernel modes for the VHS model is implemented only for dimension from 1 to 3: dimension={dimension}"
        )

    func_kwargs: dict[str, object] = {
        'num_grid':      num_grid,
        'v_max':         v_max,
        'vhs_coeff':     vhs_coeff,
        'vhs_alpha':     vhs_alpha,
        'diagonal_only': diagonal_only,
        'dtype':         dtype,
        'device':        device,
    }

    return integration_guass_legendre(
        num_roots   = num_roots,
        a           = 0.0,
        b           = 1.0,
        func        = func,
        func_kwargs = func_kwargs,
        dtype       = dtype,
        device      = device,
    )


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()