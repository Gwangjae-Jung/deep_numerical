from    typing      import  Optional, Sequence, Union
import  torch


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
]


##################################################
##################################################
# FFT frequencies
def fft_index(n: int) -> torch.Tensor:
    """Return the 1-dimensional array of all possible entries in a frequency in DFT."""
    return torch.concatenate((torch.arange((n+1) // 2), torch.arange(-(n//2), 0)))    


def freq_tensor(
        dim:        int,
        num_grid:   Union[int, Sequence[int]],
        keepdim:    bool = False
    ) -> torch.Tensor:
    """Return the collection of all possible frequencies in DFT."""
    freqs: torch.Tensor = torch.stack(
        torch.meshgrid(
            *(fft_index(num_grid) for _ in range(dim)),
            indexing = 'ij',
        ),
        axis = -1
    )
    if keepdim:
        return freqs
    else:
        return freqs.reshape(-1, dim)
        

def freq_pair_tensor(
        dim:            int,
        num_grid:       int,
        keepdim:        bool = False,
        diagonal_only:  bool = False,
    ) -> torch.Tensor:
    """Return the collection of all possible pairs frequencies in DFT.
    
    -----
    ### Arguments
    1. `dim` (`int`)
        The dimension of the velocity space.
    
    2. `num_grid` (`int`)
        The number of the grids in each velocity dimension.
    
    3. `keepdim` (`bool`, default: `False`)
        Determines whether the output tensor keeps the shape of the velocity grid.
    
    4. `diagonal_only` (`bool`, default: `False`)
        Determines whether only the diagonal pairs (the self pairs) are returned.
        Below is the shape of the output shape when `keepdim==True`.
            - If `False`, then the output tensor is of shape `(*(num_grid for _ in range(2*dim)), 2*dim)`.
            - If `True`, then the output tensor is of shape `(*(num_grid for _ in range(dim)), 2*dim)`.
    """
    freq_pairs: torch.Tensor
    if diagonal_only:
        _freqs = freq_tensor(dim, num_grid, keepdims=True)
        freq_pairs = torch.concatenate((_freqs, _freqs), axis=-1)
        del(_freqs)
    else:
        freq_pairs = torch.stack(
            torch.meshgrid(
                *(fft_index(num_grid) for _ in range(2*dim)),
                indexing = 'ij',
            ),
            axis = -1,
        )
    
    if not keepdim:
        freq_pairs = freq_pairs.reshape(-1, 2*dim)
    
    return freq_pairs

    
def freq_index_tensor(
        dim:        int,
        num_grid:   int,
    ) -> torch.Tensor:
    """Return an array which contains all possible indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    return torch.arange((num_grid**2) * dim + 1)


def freq_index_pair_tensor(
        dim:            int,
        num_grid:       int,
        diagonal_only:  bool = False,
    ) -> torch.Tensor:
    """Return an array which contains all possible pairs of indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    arr = freq_index_tensor(dim, num_grid)
    if diagonal_only:
        # In this case, the second frequency index is always 0
        arr = arr.reshape(-1, 1)
        _zeros = torch.zeros_like(arr)
        return torch.stack((arr, _zeros), axis=-1)
    else:
        return torch.stack(torch.meshgrid(arr, arr, indexing='ij'), axis=-1)


def freq_slices_low(n_modes: Sequence[int]) -> tuple[tuple[slice]]:
    kernel_slices: list[tuple[slice]] = []
    for n in n_modes:
        n_front = (n+1)//2
        n_rear  = n//2
        kernel_slices.append(tuple( (slice(None, n_front), slice(-n_rear, None)) ))
    return tuple(kernel_slices)


def select_central(
        X_fft:          torch.Tensor,
        n_modes:        Sequence[int],
        dim:            Optional[Sequence[int]] = None,
        return_slices:  bool    = False,
    ) -> tuple[torch.Tensor, Optional[tuple[slice]]]:
    if dim is None:
        dim = tuple(range(-len(n_modes), 0))
    
    if len(n_modes)!=len(dim):
        raise ValueError(
            '\n'.join(
                [
                    f"Two arguments 'n_modes' and 'dim' should be of the same shape, but",
                    f"* {len(n_modes) = }",
                    f"* {len(dim)     = }",
                ]
            )
        )
    
    X_fft: torch.Tensor = torch.fft.fftshift(X_fft, dim=dim)
    idx_lower: list[int] = []
    idx_upper: list[int] = []
    for d, m in zip(dim, n_modes):
        N = X_fft.size(d)
        idx_lower.append( (N//2) - (m//2) )
        idx_upper.append( (N//2) + ((m-1)//2) )
    
    slices: list[slice] = [slice(None) for _ in range(X_fft.ndim)]
    for d, idx1, idx2 in zip(dim, idx_lower, idx_upper):
        slices[d] = slice(idx1, idx2+1)
    slices = tuple(slices)
    
    if return_slices:
        return X_fft[*slices], slices
    else:
        return X_fft[*slices], None



def truncate_modes(
        X_fft:      torch.Tensor,
        n_modes:    Sequence[int],
        is_real:    bool            = False,
    ) -> torch.Tensor:
    r"""Low-pass filter of the FFTs with respect to the $l^\infty$ norm on the frequency space.
    
    -----
    ### Description
    This function returns the truncated FFTs by truncating all high-frequency modes.
    To do so, this function utilizes the function `torch.split()` to obtain the nonnegative and negative modes of low frequencies, and concatenate these two modes.
    If `rfftn` is used (in other words, the signal is real), then one can pass `is_real=True` to conduct truncation for already halved FFTs.
    
    -----
    ### Note
    1. Users should align `X_fft` in the following order of dimensions: `(batch, ..., dim_1, ..., dim_d, channels)`
    """
    # Determine how to truncate the FFTs
    dimension   = len(n_modes)
    n_modes_pos = [(n+1)//2 for n in n_modes]
    n_modes_neg = [n//2     for n in n_modes]
    if is_real: # real FFTs
        n_modes_neg[-1] = 0
    
    # Split and concatenate the FFTs
    for d, n_pos, n_neg in zip(range(dimension), n_modes_pos, n_modes_neg):
        _dim = -1-dimension+d
        if X_fft.shape[_dim] < n_modes[d]:
            # No truncation if the required number of modes is less than the size
            continue
        redundant_size = X_fft.size(_dim) - (n_pos + n_neg)
        X_fft_pos, _, X_fft_neg = torch.split(X_fft, [n_pos, redundant_size, n_neg], dim=_dim)
        X_fft = torch.concatenate((X_fft_pos, X_fft_neg), dim=_dim)
    
    # Return the truncated FFTs
    return X_fft


##################################################
##################################################
# FFT operations
## Convolutions using FFT
def convolve_signals(
        x1:     torch.Tensor,
        x2:     torch.Tensor,
        dim:    Optional[Sequence[int]] = None
    ) -> torch.Tensor:
    """Computes the convolution of two input signals.
    
    -----
    ### Description
    This function computes the convolution of two (discrete) input siganls `x1` and `x2` by leveraging the convolution theorem.
    
    -----
    ### Note
    As the operation held on the frequency domain is just the pointwise multiplication, the output does not suffer from aliasing.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    if dim is None:
        dim = tuple((v for v in range(x1.ndim)))
    _norm = 'forward'
    x1_fft: torch.Tensor = torch.fft.fftn(x1, dim=dim, norm=_norm)
    x2_fft: torch.Tensor = torch.fft.fftn(x2, dim=dim, norm=_norm)
    conv_fft = x1_fft * x2_fft
    conv: torch.Tensor = torch.fft.ifftn(conv_fft, dim=dim, norm=_norm)
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
    if x1_fft.shape != x2_fft.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1_fft.shape: {x1_fft.shape}\n"
            f"* x2_fft.shape: {x2_fft.shape}\n"
        )
    if dim is None:
        dim = tuple((v for v in range(x1_fft.ndim)))
    _norm = 'forward'
    x1: torch.Tensor = torch.fft.ifftn(x1_fft, dim=dim, norm=_norm)
    x2: torch.Tensor = torch.fft.ifftn(x2_fft, dim=dim, norm=_norm)
    return torch.fft.fftn(x1*x2, dim=dim, norm=_norm)


##################################################
##################################################
# End of file