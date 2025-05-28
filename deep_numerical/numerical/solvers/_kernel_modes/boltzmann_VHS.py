import  warnings
from    typing              import  Callable, Optional

import  torch
from    torch.special       import  bessel_j0   as  j0

from    ....     import  utils


##################################################
##################################################
__all__: list[str] = [
    # Spectral method
    'Boltzmann_VHS_kernel_modes_1D_integrand',
    'Boltzmann_VHS_kernel_modes_2D_integrand',
    'Boltzmann_VHS_kernel_modes_3D_integrand',
    'Boltzmann_VHS_kernel_modes',
]


##################################################
##################################################
# Spectral method
## The kernel components (1-dimensional)
def Boltzmann_VHS_kernel_modes_1D_integrand(
        r:          torch.Tensor,
    
        num_grid:   int,
        v_max:      float,
        vhs_coeff:  float,
        vhs_alpha:  float,
        
        diagonal_only:  bool    = False,
        
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 1.
    """
    dimension = 1
    if diagonal_only:
        _r = r.reshape(*utils.ones(dimension), -1)
    else:
        _r = r.reshape(*utils.ones(2*dimension), -1)
    _prod1 = \
        vhs_coeff * \
        ((2*utils.LAMBDA*v_max)**(dimension+vhs_alpha)) * \
        torch.pow(_r, (dimension-1)+vhs_alpha)
    
    # Compute `r_freq_1` and `r_freq_2`, the scales which are applied to `r`
    freq_pair = utils.freq_pair_tensor(
                        dimension       = dimension,
                        num_grid        = num_grid,
                        keepdim        = True,
                        diagonal_only   = diagonal_only,
                        dtype           = dtype,
                        device          = device,
                    )
    freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
    r_freq1 = (utils.LAMBDA*torch.pi) * torch.norm(
                    freq_pair_1+freq_pair_2,
                    p=2, dim=-1, keepdim=True,
                )
    r_freq2 = (utils.LAMBDA*torch.pi) * torch.norm(
                    freq_pair_1-freq_pair_2,
                    p=2, dim=-1, keepdim=True,
                )
    del(freq_pair_1, freq_pair_2, freq_pair)
    
    _prod2a = 2 * torch.cos(r_freq1 * _r)  # <-- Scale modification (Done: 1D)
    _prod2b = 2 * torch.cos(r_freq2 * _r)  # <-- Scale modification (Done: 1D)
    _prod2  = _prod2a * _prod2b
    
    return _prod1 * _prod2


## The kernel components (2-dimensional)    
def Boltzmann_VHS_kernel_modes_2D_integrand(
        r:              torch.Tensor,
    
        num_grid:       int,
        v_max:          float,
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        diagonal_only:  bool    = False,
        
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 2.
    """
    dimension = 2
    if diagonal_only:
        _r = r.reshape(*utils.ones(dimension), -1)
    else:
        _r = r.reshape(*utils.ones(2*dimension), -1)
    _prod1 = \
        vhs_coeff * \
        ((2*utils.LAMBDA*v_max)**(dimension+vhs_alpha)) * \
        torch.pow(_r, (dimension-1)+vhs_alpha)
    
    # Compute `r_freq_1` and `r_freq_2`, the scales which are applied to `r`
    freq_pair = utils.freq_pair_tensor(
                        dimension       = dimension,
                        num_grid        = num_grid,
                        keepdim        = True,
                        diagonal_only   = diagonal_only,
                        dtype           = dtype,
                        device          = device,
                    )
    freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
    r_freq1 = (utils.LAMBDA*torch.pi) * torch.norm(
                    freq_pair_1+freq_pair_2,
                    p=2, dim=-1, keepdim=True,
                )
    r_freq2 = (utils.LAMBDA*torch.pi) * torch.norm(
                    freq_pair_1-freq_pair_2,
                    p=2, dim=-1, keepdim=True,
                )
    del(freq_pair_1, freq_pair_2, freq_pair)
    
    _prod2a = (2*torch.pi) * j0(r_freq1 * _r)  # <-- Scale modification (Done: 2D)
    _prod2b = (2*torch.pi) * j0(r_freq2 * _r)  # <-- Scale modification (Done: 2D)
    _prod2  = _prod2a * _prod2b
    
    return _prod1 * _prod2
    

## The kernel components (3-dimensional)
def Boltzmann_VHS_kernel_modes_3D_integrand(
        r:              torch.Tensor,
    
        num_grid:       int,
        v_max:          float,
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        diagonal_only:  bool    = False,
        
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """The integrand of the integral defining the kernel modes of the VHS model for dimension 3.
    
    -----
    ### Note
    When `diagonal_only == False` (where all kernel modes are required), this function returns an efficient storage to call the values of the kernel modes, which is a 2-tensor of shape `(3*(num_grid**2), 3*(num_grid**2))`.
    Hence, given two 3-dimensional (integer-valued) FFT indices `freq1` and `freq2`, to call the kernel mode corresponding to the pair `(freq1, freq2)` from the output tensor `ret`,
        * unlike it is called as `ret[*tuple(freq1), *tuple(freq2)]` for `dimension < 3`,
        * in the 3-dimensional case, one should precompute the square of the 2-norms `a` and `b` of `freq1+freq2` and `freq1-freq2`, respectively, and call the kernel mode as `ret[a, b]`.
    """
    ##################################################
    # NOTE: The efficient implementation is done ONLY FOR `diagonal_only==False`.
    ##################################################
    dimension = 3
    if diagonal_only:
        _r = r.reshape(*utils.ones(dimension), -1)
    else:
        _r = r.reshape(*utils.ones(2), -1)    # <-- Efficient implementation
    _prod1 = \
        vhs_coeff * \
        ((2*utils.LAMBDA*v_max)**(dimension+vhs_alpha)) * \
        torch.pow(_r, (dimension-1)+vhs_alpha)
        
    # Compute `r_freq_1` and `r_freq_2`, the scales which are applied to `r`
    if diagonal_only:
        freq_pair = utils.freq_pair_tensor(
                        dimension       = dimension,
                        num_grid        = num_grid,
                        keepdim        = True,
                        diagonal_only   = True,
                        dtype           = dtype,
                        device          = device,
                    )
        freq_pair_1, freq_pair_2 = freq_pair[..., :dimension], freq_pair[..., dimension:]
        r_freq1 = (utils.LAMBDA*torch.pi) * torch.norm(
                        freq_pair_1+freq_pair_2,
                        p=2, dim=-1, keepdim=True,
                    )
        r_freq2 = (utils.LAMBDA*torch.pi) * torch.norm(
                        freq_pair_1-freq_pair_2,
                        p=2, dim=-1, keepdim=True,
                    )
        del(freq_pair_1, freq_pair_2, freq_pair)
        _prod2a = (4*torch.pi) * utils.sinc(r_freq1 * _r)  # <-- Scale modification (Done: 3D)
        _prod2b = (4*torch.pi) * utils.sinc(r_freq2 * _r)  # <-- Scale modification (Done: 3D)
        _prod2  = _prod2a * _prod2b
        
    else:
        idx_pair = utils.freq_index_pair_tensor(
                        dimension       = dimension,
                        num_grid        = num_grid,
                        diagonal_only   = False,
                    ).astype(torch.float64)
        r_freq1 = (utils.LAMBDA*torch.pi) * torch.sqrt(idx_pair[..., [0]])  # A 3-tensor
        r_freq2 = (utils.LAMBDA*torch.pi) * torch.sqrt(idx_pair[..., [1]])  # A 3-tensor
        del(idx_pair)
        _prod2a = (4*torch.pi) * utils.sinc(r_freq1 * _r)    # <-- Scale modification (Done: 3D)
        _prod2b = (4*torch.pi) * utils.sinc(r_freq2 * _r)    # <-- Scale modification (Done: 3D)
        _prod2  = _prod2a * _prod2b
        
    return _prod1 * _prod2


def Boltzmann_VHS_kernel_modes(
        dimension:  int,
        num_grid:   int,
        
        v_max:      float,
        vhs_coeff:  float,
        vhs_alpha:  float,
        
        num_roots:      Optional[int] = None,
        diagonal_only:  bool = False,
        
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Computes the kernel modes for a given set of configurations.
    
    -----
    ### Note
    1. By the kernel mode in this function we do not mean the scale of the product of two Fourier coefficients which defines the time-derivative of the Fourier coefficients.
    To compute the scale for the pair `(l, m)` of two multiindices `l` and `m`, one should compute `ret[tuple(l), tuple(m)] - ret[tuple(m), tuple(m)]`, where `ret` is the output of this function.
    """
    # NOTE: The efficient implementation is done ONLY FOR the case where `dim==3` and `diagonal_only==False`.    
    # Set the order of the Gauss-Legendre quadrature rule if `num_roots` is not set
    if num_roots is None:
        num_roots = num_grid
    
    # Set the integrand
    func: Callable[[torch.Tensor, dict[str, object]], torch.Tensor]
    if dimension==1:
        func = Boltzmann_VHS_kernel_modes_1D_integrand
    elif dimension==2:
        func = Boltzmann_VHS_kernel_modes_2D_integrand
    elif dimension==3:
        warnings.warn(
            ' '.join(
                [
                    f"For efficient implementation, the kernel modes for the 3-dimensional case should be referred by computing the 2-norms of the addition and subtraction of two input frequencies.",
                    f"Use the method 'compute_integral_indices' of this class."
                ]
            ),
            Warning
        )
        func = Boltzmann_VHS_kernel_modes_3D_integrand
    else:
        raise ValueError(
            ' '.join(
                [
                    f"The computation of the kernel modes for the VHS model is implemented only for dimension from 1 to 3.",
                    f"({dimension=})",
                ]
            )
        )
    func_kwargs: dict[str, object] = {
        'num_grid':         num_grid,
        'v_max':            v_max,
        'vhs_coeff':        vhs_coeff,
        'vhs_alpha':        vhs_alpha,
        'diagonal_only':    diagonal_only,
        'dtype':            dtype,
        'device':           device,
    }
        
    # Return the result
    return utils.integration_guass_legendre(
        num_roots   = num_roots,
        a           = 0.0,
        b           = 1.0,
        func        = func,
        func_kwargs = func_kwargs,
        dtype       = dtype,
        device      = device,
    )


##################################################
##################################################
# End of file