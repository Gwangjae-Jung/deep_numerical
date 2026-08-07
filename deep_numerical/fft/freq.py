from    typing      import  Optional, Sequence, Union
from    itertools   import  product
import  torch
from    deep_numerical  import  repeat


__all__ = [
    # FFT frequencies
    'fft_index',
    'freq_tensor',
    'freq_pair_tensor',
    'freq_index_tensor',
    'freq_index_pair_tensor',
    'freq_slices_low',
    
    # FFT utils
    'fft_prod_slices',
    'fft_compression',
    'fft_expansion',
]


##################################################
##################################################
# FFT frequencies
def fft_index(
        n:      int,
        dtype:  torch.dtype             = torch.long,
        device: Optional[torch.device]  = None,
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
        
        dtype:      torch.dtype             = torch.long,
        device:     Optional[torch.device]  = None,
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
        device:         Optional[torch.device]  = None,
    ) -> torch.Tensor:
    """Return the collection of all possible pairs frequencies in DFT.
    
    Arguments:
        `dimension` (`int`): The dimension of the velocity space.
        `num_grid` (`int`): The number of the grids in each velocity dimension.
        `keepdim` (`bool`, default: `False`): Determines whether the output tensor keeps the shape of the velocity grid.
        `diagonal_only` (`bool`, default: `False`): Determines whether only the diagonal pairs (the self pairs) are returned. Below is the shape of the output shape when `keepdims==True`.
            - If `False`, then the output tensor is of shape `(*repeat(num_grid, 2*dimension)), 2*dimension)`.
            - If `True`, then the output tensor is of shape `(*repeat(num_grid, dimension), 2*dimension)`.    
        `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
        `device` (`torch.device`, default: `None`): The device of the output tensor.
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
        device:     Optional[torch.device]  = None,
    ) -> torch.Tensor:
    """Return an array which contains all possible indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`."""
    return torch.arange((num_grid**2) * dimension + 1, dtype=dtype, device=device)


def freq_index_pair_tensor(
        dimension:      int,
        num_grid:       int,
        diagonal_only:  bool = False,
        
        dtype:          torch.dtype     = torch.long,
        device:     Optional[torch.device]  = None,
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
# FFT utils
def fft_prod_slices(ndim: int, dim: Sequence[int], n_modes: Sequence[int]) -> product:
    """Return the product of slices for the FFT operations acted on low-frequency modes.
    
    Arguments:
        `ndim` (`int`):
            The number of dimensions of the input tensor.
        `dim` (`Sequence[int]`):
            The dimensions on which the FFT operations are acted.
        `n_modes` (`Sequence[int]`):
            The number of modes in each dimension to be preserved.
    
    Returns:
        `product`: A product of slices, where each slice corresponds to the low-frequency modes in the specified dimensions
    """
    _slices: list[Sequence[slice]] = [tuple([slice(None)])] * ndim
    for d, n in zip(dim, n_modes):
        n_front = (n+1)//2
        n_rear  = n//2
        _slices[d] = tuple(( slice(None, n_front) , slice(-n_rear, None) ))
    return product(*_slices)


def fft_compression(
        X_fft:              torch.Tensor,
        dim:                Sequence[int],
        compression_size:   Sequence[int],
    ) -> torch.Tensor:
    """Compress the FFT of a low-resolutional signal to the corresponding high-resolutional FFT.
    
    Arguments:
        `X_fft` (`torch.Tensor`):
            The input tensor in the frequency domain, which is the FFT of a low-resolutional signal.
        `dim` (`Sequence[int]`):
            The dimensions on which the compression is applied.
        `compression_size` (`Sequence[int]`):
            The sizes of the compressed dimensions.
    
    Returns:
        `torch.Tensor`: The complex tensor, where only the low-frequency modes of `X_fft` are preserved. The shape of this tensor is not same as `X_fft`, but it is the same as the shape of the low-frequency modes.
    """
    if len(dim) != len(compression_size):
        raise ValueError(
            f"The length of 'dim' ({len(dim)}) and 'compression_size' ({len(compression_size)}) should be the same."
        )
    ndim = X_fft.ndim
    # Adjust the dimensions to be within `range(X_fft.ndim)`
    dim = tuple((d % ndim for d in dim))
    for d, size_c in zip(dim, compression_size):
        size_x = X_fft.size(d)
        if size_x < size_c:
            raise ValueError(
                f"At dimension {d}, the input tensor is of size {size_x}, while the compression size is {size_c}."
            )
    
    newshape = list(X_fft.shape)
    for d, s in zip(dim, compression_size):
        newshape[d] = s
    
    Y_fft = torch.zeros(list(newshape), dtype=X_fft.dtype, device=X_fft.device)
    for sl in fft_prod_slices(ndim, dim, compression_size):
        Y_fft[*sl] = X_fft[*sl]
    return Y_fft


def fft_expansion(
        X_fft:          torch.Tensor,
        dim:            Sequence[int],
        expansion_size: Sequence[int],
    ) -> torch.Tensor:
    """Expand the FFT of a low-resolutional signal to the corresponding high-resolutional FFT.
    
    Arguments:
        `X_fft` (`torch.Tensor`):
            The input tensor in the frequency domain, which is the FFT of a low-resolutional signal.
        `dim` (`Sequence[int]`):
            The dimensions on which the expansion is applied.
        `expansion_size` (`Sequence[int]`):
            The sizes of the expanded dimensions.
    
    Returns:
        `torch.Tensor`: The complex tensor, where the low-frequency modes of `X_fft` are expanded to the high-frequency modes. The shape of this tensor is not same as `X_fft`, but it is the same as the shape of the high-frequency modes.
    """
    if len(dim) != len(expansion_size):
        raise ValueError(
            f"The length of 'dim' ({len(dim)}) and 'expansion_size' ({len(expansion_size)}) should be the same."
        )
    ndim = X_fft.ndim
    # Adjust the dimensions to be within `range(X_fft.ndim)`
    dim = tuple((d % ndim for d in dim))
    for d, size_e in zip(dim, expansion_size):
        size_x = X_fft.size(d)
        if size_x > size_e:
            raise ValueError(
                f"At dimension {d}, the input tensor is of size {size_x}, while the expansion size is {size_e}."
            )
    
    newshape = list(X_fft.shape)
    for d, s in zip(dim, expansion_size):
        newshape[d] = s
    n_modes = tuple([X_fft.size(d) for d in dim])
    
    Y_fft = torch.zeros(list(newshape), dtype=X_fft.dtype, device=X_fft.device)
    for sl in fft_prod_slices(ndim, dim, n_modes):
        Y_fft[*sl] = X_fft[*sl]
    return Y_fft


##################################################
##################################################
# End of file