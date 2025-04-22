from    typing              import  Optional, Sequence, Union
import  numpy               as      np


##################################################
##################################################
__all__: list[str] = [
    # FFT frequencies
    'fft_index',
    'freq_tensor',
    'freq_pair_tensor',
    'freq_index_tensor',
    'freq_index_pair_tensor',
    
    # FFT operations
    'convolve_signals',
    'convolve_freqs',
    'linear_convolution',
    'circular_convolution',
    'truncated_convolution',
]


##################################################
##################################################
# FFT frequencies
def fft_index(n: int) -> np.ndarray:
    """Return the 1-dimensional array of all possible entries in a frequency in DFT."""
    return np.concatenate((np.arange((n+1) // 2), np.arange(-(n//2), 0)))    


def freq_tensor(
        dim:        int,
        num_grid:   Union[int, Sequence[int]],
        keepdims:   bool = False
    ) -> np.ndarray:
    """Return the collection of all possible frequencies in DFT."""
    freqs = np.stack(
        np.meshgrid(
            *(fft_index(num_grid) for _ in range(dim)),
            indexing = 'ij',
        ),
        axis = -1
    )
    if keepdims:
        return freqs
    else:
        return freqs.reshape(-1, dim)
        

def freq_pair_tensor(
        dim:            int,
        num_grid:       int,
        keepdims:       bool = False,
        diagonal_only:  bool = False,
    ) -> np.ndarray:
    """Return the collection of all possible pairs frequencies in DFT.
    
    -----
    ### Arguments
    1. `dim` (`int`)
        The dimension of the velocity space.
    
    2. `num_grid` (`int`)
        The number of the grids in each velocity dimension.
    
    3. `keepdims` (`bool`, default: `False`)
        Determines whether the output tensor keeps the shape of the velocity grid.
    
    4. `diagonal_only` (`bool`, default: `False`)
        Determines whether only the diagonal pairs (the self pairs) are returned.
        Below is the shape of the output shape when `keepdims==True`.
            - If `False`, then the output tensor is of shape `(*(num_grid for _ in range(2*dim)), 2*dim)`.
            - If `True`, then the output tensor is of shape `(*(num_grid for _ in range(dim)), 2*dim)`.
    """
    freq_pairs: np.ndarray
    if diagonal_only:
        _freqs = freq_tensor(dim, num_grid, keepdims=True)
        freq_pairs = np.concatenate((_freqs, _freqs), axis=-1)
        del(_freqs)
    else:
        freq_pairs = np.stack(
            np.meshgrid(
                *(fft_index(num_grid) for _ in range(2*dim)),
                indexing = 'ij',
            ),
            axis = -1,
        )
    
    if not keepdims:
        freq_pairs = freq_pairs.reshape(-1, 2*dim)
    
    return freq_pairs

    
def freq_index_tensor(
        dim:        int,
        num_grid:   int,
    ) -> np.ndarray:
    """Return an array which contains all possible indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    return np.arange((num_grid**2) * dim + 1)


def freq_index_pair_tensor(
        dim:            int,
        num_grid:       int,
        diagonal_only:  bool = False,
    ) -> np.ndarray:
    """Return an array which contains all possible pairs of indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    arr = freq_index_tensor(dim, num_grid)
    if diagonal_only:
        # In this case, the second frequency index is always 0
        arr = arr.reshape(-1, 1)
        _zeros = np.zeros_like(arr)
        return np.stack((arr, _zeros), axis=-1)
    else:
        return np.stack(np.meshgrid(arr, arr, indexing='ij'), axis=-1)


##################################################
##################################################
# FFT operations
## Convolutions using FFT
def convolve_signals(
        x1:     np.ndarray,
        x2:     np.ndarray,
        axes:   Optional[Sequence[int]] = None
    ) -> np.ndarray:
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
    if axes is None:
        axes = tuple((v for v in range(x1.ndim)))
    _norm = 'forward'
    x1_fft = np.fft.fftn(x1, axes=axes, norm=_norm)
    x2_fft = np.fft.fftn(x2, axes=axes, norm=_norm)
    conv_fft = x1_fft * x2_fft
    conv = np.fft.ifftn(conv_fft, axes=axes, norm=_norm)
    if np.iscomplexobj(x1) or np.iscomplexobj(x2):
        return conv
    else:
        return conv.real


def convolve_freqs(
        x1_fft: np.ndarray,
        x2_fft: np.ndarray,
        axes:   Optional[Sequence[int]] = None,
    ) -> np.ndarray:
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
    if axes is None:
        axes = tuple((v for v in range(x1_fft.ndim)))
    _norm = 'forward'
    x1 = np.fft.ifftn(x1_fft, axes=axes, norm=_norm)
    x2 = np.fft.ifftn(x2_fft, axes=axes, norm=_norm)
    return np.fft.fftn(x1 * x2, axes=axes, norm=_norm)


def circular_convolution(
        x1:     np.ndarray,
        x2:     np.ndarray,
        axes:   Optional[Sequence[int]] = None,
    ) -> np.ndarray:
    """Returns the circular convolution of two input arrays.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    shape = x1.shape
    
    if axes is None:
        axes = tuple((v for v in range(x1.ndim)))
    else:
        axes = tuple(axes)
    
    a_fft = np.fft.fftn(x1, s=shape, axes=axes)
    b_fft = np.fft.fftn(x2, s=shape, axes=axes)
    
    u_fft = a_fft * b_fft
    u = np.fft.ifftn(u_fft, s=shape, axes=axes)
    if np.iscomplexobj(x1) or np.iscomplexobj(x2):
        return u
    else:
        return u.real


def linear_convolution(
        x1:     np.ndarray,
        x2:     np.ndarray,
        axes:   Optional[Sequence[int]] = None,
    ) -> np.ndarray:
    """Returns the linear convolution of two input arrays.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    
    if axes is None:
        axes = tuple((v for v in range(x1.ndim)))
    else:
        axes = tuple(axes)
    
    shape_conv = tuple((x1.shape[_axis] + x2.shape[_axis] - 1 for _axis in axes))
    a_fft = np.fft.fftn(x1, s=shape_conv, axes=axes)
    b_fft = np.fft.fftn(x2, s=shape_conv, axes=axes)
    
    u_fft = a_fft * b_fft
    u = np.fft.ifftn(u_fft, s=shape_conv, axes=axes)
    if np.iscomplexobj(x1) or np.iscomplexobj(x2):
        return u
    else:
        return u.real


def truncated_convolution(
        x1:             np.ndarray,
        x2:             np.ndarray,
        axes:           Optional[Sequence[int]] = None,
        fft_indexed:    bool    = True,
    ) -> np.ndarray:
    """Returns the truncated convolution of two input arrays.
    
    -----
    ### Implementation
    Assume `fft_indexed == True`.
    1. Shift `x1` and `x2`, and denote them by `a1` and `a2`, respectively.
    2. Compute the linear convolution `z` of `a1` and `a2`.
    3. Discard the entries of `z` belonging to non-allowed frequencies.
    4. Inversely shift `z` to return as the .
    
    -----
    ### Note
    1. The argument `fft_indexed` suggests whether the input arrays `x1` and `x2` are aligned in accordance with the usual indexing for the FFTs along `axes`.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {x1.shape}\n"
            f"* x2.shape: {x2.shape}\n"
        )
    
    if axes is None:
        axes = tuple((v for v in range(x1.ndim)))
    else:
        axes = tuple(axes)
    
    # Step 1. Shift the input arrays
    a1 = np.fft.fftshift(x1, axes=axes) if fft_indexed else x1
    a2 = np.fft.fftshift(x2, axes=axes) if fft_indexed else x2
    
    # Step 2. Compute the linear convolution
    z = linear_convolution(a1, a2, axes)
    
    # Step 3. Discard non-allowed entries
    _slices = []
    for cnt in range(x1.ndim):
        if cnt in axes:
            _n_curr = x1.shape[cnt]
            _left   = +(_n_curr // 2)
            _right  = -((_n_curr-1) // 2)
            _slices.append(slice(_left, _right))
        else:
            _slices.append(slice(None))
    z = z[tuple(_slices)]
    
    # Step 4. Inverse shift
    z = np.fft.ifftshift(z, axes=axes)  # <-- DO NOT FORGET TO WRITE DOWN ALL ARGUMENTS
    
    # Return the result
    if np.iscomplexobj(x1) or np.iscomplexobj(x2):
        return z
    else:
        return z.real


##################################################
##################################################
# End of file