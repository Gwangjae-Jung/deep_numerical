from    typing      import  Optional, Sequence, Union
import  torch
from    ._dtype     import  TORCH_DEFAULT_DEVICE
from    ._helper    import  repeat


##################################################
##################################################
__all__: list[str] = [
    # FFT frequencies
    'fft_index',
    'freq_tensor',
    'freq_pair_tensor',
    'freq_index_tensor',
    'freq_index_pair_tensor',
    'freq_slices_low',
    
    # FFT operations
    'convolve_signals',
    'convolve_freqs',
    'linear_convolution',
    'circular_convolution',
]


##################################################
##################################################
FFT_NORM:   str = 'forward'


##################################################
##################################################
# FFT frequencies
def fft_index(
        n:      int,
        
        dtype:  torch.dtype     = torch.long,
        device: torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return the 1-dimensional array of all possible entries in a frequency in DFT."""
    return torch.concatenate(
        (
            torch.arange((n+1)//2, dtype=dtype, device=device),
            torch.arange(-(n//2), 0, dtype=dtype, device=device),
        )
    )


def freq_tensor(
        dimension:  int,
        num_grid:   Union[int, Sequence[int]],
        keepdim:    bool = False,
        
        dtype:      torch.dtype     = torch.long,
        device:     torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return the collection of all possible frequencies in DFT."""
    freqs: torch.Tensor = torch.stack(
        torch.meshgrid(
            *repeat(fft_index(num_grid, dtype, device), dimension),
            indexing = 'ij',
        ),
        dim = -1
    )
    if keepdim:
        return freqs
    else:
        return freqs.reshape(-1, dimension)
        

def freq_pair_tensor(
        dimension:      int,
        num_grid:       int,
        keepdim:        bool = False,
        diagonal_only:  bool = False,
        
        dtype:          torch.dtype     = torch.long,
        device:         torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return the collection of all possible pairs frequencies in DFT.
    
    -----
    ### Arguments
    * `dimension` (`int`)
        * The dimension of the velocity space.
    
    * `num_grid` (`int`)
        * The number of the grids in each velocity dimension.
        
    * `keepdim` (`bool`, default: `False`)
        * Determines whether the output tensor keeps the shape of the velocity grid.
    
    * `diagonal_only` (`bool`, default: `False`)
        * Determines whether only the diagonal pairs (the self pairs) are returned. Below is the shape of the output shape when `keepdims==True`.
            - If `False`, then the output tensor is of shape `(*repeat(num_grid, 2*dimension)), 2*dimension)`.
            - If `True`, then the output tensor is of shape `(*repeat(num_grid, dimension), 2*dimension)`.
    
    * `dtype` (`torch.dtype`, default: `torch.long`)
        * The data type of the output tensor.
    
    * `device` (`torch.device`, default: `TORCH_DEFAULT_DEVICE`)
        * The device of the output tensor.
    """
    freq_pairs: torch.Tensor
    if diagonal_only:
        _freqs = freq_tensor(dimension, num_grid, keepdim=True, dtype=dtype, device=device)
        freq_pairs = torch.concatenate((_freqs, _freqs), dim=-1)
        del(_freqs)
    else:
        freq_pairs = torch.stack(
            torch.meshgrid(
                *repeat(fft_index(num_grid, dtype, device), 2*dimension),
                indexing = 'ij',
            ),
            dim = -1,
        )
    if not keepdim:
        freq_pairs = freq_pairs.reshape(-1, 2*dimension)
    return freq_pairs

    
def freq_index_tensor(
        dimension:  int,
        num_grid:   int,
        
        dtype:      torch.dtype     = torch.long,
        device:     torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return an array which contains all possible indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    return torch.arange((num_grid**2) * dimension + 1, dtype=dtype, device=device)


def freq_index_pair_tensor(
        dimension:      int,
        num_grid:       int,
        diagonal_only:  bool = False,
        
        dtype:          torch.dtype     = torch.long,
        device:         torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return an array which contains all possible pairs of indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    arr = freq_index_tensor(dimension, num_grid, dtype=dtype, device=device)
    if diagonal_only:
        # In this case, the second frequency index is always 0
        arr = arr.reshape(-1, 1)
        _zeros = torch.zeros_like(arr, dtype=dtype, device=device)
        return torch.stack((arr, _zeros), dim=-1)
    else:
        return torch.stack(torch.meshgrid(arr, arr, indexing='ij'), dim=-1)


def freq_slices_low(n_modes: Sequence[int]) -> tuple[tuple[slice]]:
    kernel_slices: list[tuple[slice]] = []
    for n in n_modes:
        n_front = (n+1)//2
        n_rear  = n//2
        kernel_slices.append(tuple( (slice(None, n_front), slice(-n_rear, None)) ))
    return tuple(kernel_slices)


##################################################
##################################################
# FFT operations
## Convolutions using FFT
def convolve_signals(
        x1:     torch.Tensor,
        x2:     torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Computes the convolution of two input signals.
    
    -----
    ### Description
    This function computes the convolution of two (discrete) input siganls `x1` and `x2` by leveraging the convolution theorem.
    
    -----
    ### Note
    As the operation held on the frequency domain is just the pointwise multiplication, the output does not suffer from aliasing.
    """
    global FFT_NORM
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    if dim is None:
        dim = tuple((v for v in range(x1.ndim)))
    else:
        dim = tuple(dim)
    x1_fft = torch.fft.fftn(x1, dim=dim, norm=FFT_NORM)
    x2_fft = torch.fft.fftn(x2, dim=dim, norm=FFT_NORM)
    conv_fft = x1_fft*x2_fft
    conv: torch.Tensor = torch.fft.ifftn(conv_fft, dim=dim, norm=FFT_NORM)
    if torch.is_complex(x1) or torch.is_complex(x2):
        return conv
    else:
        return conv.real


def convolve_freqs(
        x1_fft: torch.Tensor,
        x2_fft: torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Computes the convolution of two input FFTs.
    
    -----
    ### Description
    This function computes the convolution of two (discrete) input FFTs `x1_fft` and `x2_fft` by leveraging the fact that the Fourier coefficients of the product of two signals is the convolution of the Fourier coefficients.
    
    -----
    ### Note
    The output of this function is valid only for the *finite* signals.
    Hence, the output may not be used to recover the multiplication of the original signals, unless the sampling frequency is not less than the Nyquist frequency.
    """
    global FFT_NORM
    if x1_fft.shape != x2_fft.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1_fft.shape: {x1_fft.shape}\n"
            f"* x2_fft.shape: {x2_fft.shape}\n"
        )
    if dim is None:
        dim = tuple((v for v in range(x1_fft.ndim)))
    else:
        dim = tuple(dim)
    x1 = torch.fft.ifftn(x1_fft, dim=dim, norm=FFT_NORM)
    x2 = torch.fft.ifftn(x2_fft, dim=dim, norm=FFT_NORM)
    return torch.fft.fftn(x1*x2, dim=dim, norm=FFT_NORM)


def circular_convolution(
        x1:     torch.Tensor,
        x2:     torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Returns the circular convolution of two input arrays.
    """
    global FFT_NORM
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    shape = x1.shape
    if dim is None:
        dim = tuple((v for v in range(x1.ndim)))
    else:
        dim = tuple(dim)
    
    a_fft = torch.fft.fftn(x1, s=shape, dim=dim, norm=FFT_NORM)
    b_fft = torch.fft.fftn(x2, s=shape, dim=dim, norm=FFT_NORM)
    
    u_fft = a_fft * b_fft
    u: torch.Tensor = torch.fft.ifftn(u_fft, s=shape, dim=dim, norm=FFT_NORM)
    if torch.is_complex(x1) or torch.is_complex(x2):
        return u
    else:
        return u.real


def linear_convolution(
        x1:     torch.Tensor,
        x2:     torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Returns the linear convolution of two input arrays.
    """
    global FFT_NORM
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    if dim is None:
        dim = tuple((v for v in range(x1.ndim)))
    else:
        dim = tuple(dim)
    
    shape_conv = tuple((x1.shape[_dim] + x2.shape[_dim] - 1 for _dim in dim))
    a_fft = torch.fft.fftn(x1, s=shape_conv, dim=dim, norm=FFT_NORM)
    b_fft = torch.fft.fftn(x2, s=shape_conv, dim=dim, norm=FFT_NORM)
    
    u_fft = a_fft * b_fft
    u: torch.Tensor = torch.fft.ifftn(u_fft, s=shape_conv, dim=dim, norm=FFT_NORM)
    if torch.is_complex(x1) or torch.is_complex(x2):
        return u
    else:
        return u.real


##################################################
##################################################
# End of file