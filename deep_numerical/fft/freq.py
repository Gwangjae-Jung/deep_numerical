from   itertools import product
from   typing    import Optional, Sequence, Union
import torch

from   deep_numerical import repeat


__all__: list[str] = [
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
def fft_index(
    n:      int,
    dtype:  torch.dtype            = torch.long,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return the 1-dimensional array of all possible entries in a frequency in DFT.

    ## Description
    Returns the 1-dimensional array of all possible entries in a frequency in discrete Fourier transform (DFT).

    ## Arguments
    `n` (`int`): The number of grid points.
    `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensor.

    ## Returns
    `torch.LongTensor`: A 1-dimensional tensor of frequency indices.
    """
    return torch.concatenate(
        (
            torch.arange((n + 1) // 2, dtype=dtype, device=device),
            torch.arange(-(n // 2), 0, dtype=dtype, device=device),
        )
    )


def freq_tensor(
    dimension: int,
    num_grid:  Union[int, Sequence[int]],
    keepdim:   bool                   = False,
    dtype:     torch.dtype            = torch.long,
    device:    Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return the collection of all possible frequencies in DFT.

    ## Description
    Returns the tensor collection of all possible multi-dimensional frequencies in discrete Fourier transform (DFT).

    ## Arguments
    `dimension` (`int`): The spatial dimension.
    `num_grid` (`Union[int, Sequence[int]]`): The number of grid points in each dimension.
    `keepdim` (`bool`, default: `False`): Whether to keep the spatial grid shape or flatten into `(-1, dimension)`.
    `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensor.

    ## Returns
    `torch.LongTensor`: The tensor of frequency vectors.
    """
    freqs: torch.LongTensor = torch.stack(
        torch.meshgrid(
            *repeat(fft_index(num_grid, dtype, device), dimension),
            indexing = 'ij',
        ),
        dim = -1,
    )
    if keepdim:
        return freqs
    else:
        return freqs.reshape(-1, dimension)


def freq_pair_tensor(
    dimension:     int,
    num_grid:      int,
    keepdim:       bool                   = False,
    diagonal_only: bool                   = False,
    dtype:         torch.dtype            = torch.long,
    device:        Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return the collection of all possible pairs of frequencies in DFT.

    ## Description
    Returns the collection of all possible pairs of frequencies in discrete Fourier transform (DFT).

    ## Arguments
    `dimension` (`int`): The dimension of the velocity space.
    `num_grid` (`int`): The number of grids in each velocity dimension.
    `keepdim` (`bool`, default: `False`): Determines whether the output tensor keeps the shape of the velocity grid.
    `diagonal_only` (`bool`, default: `False`): Determines whether only the diagonal pairs (the self pairs) are returned.
    `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensor.

    ## Returns
    `torch.LongTensor`: The tensor of frequency pairs.
    """
    freq_pairs: torch.LongTensor
    if diagonal_only:
        _freqs: torch.LongTensor = freq_tensor(dimension, num_grid, keepdim=True, dtype=dtype, device=device)
        freq_pairs = torch.concatenate((_freqs, _freqs), dim=-1)
        del _freqs
    else:
        freq_pairs = torch.stack(
            torch.meshgrid(
                *repeat(fft_index(num_grid, dtype, device), 2 * dimension),
                indexing = 'ij',
            ),
            dim = -1,
        )
    if not keepdim:
        freq_pairs = freq_pairs.reshape(-1, 2 * dimension)
    return freq_pairs


def freq_index_tensor(
    dimension: int,
    num_grid:  int,
    dtype:     torch.dtype            = torch.long,
    device:    Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return an array containing all possible index values.

    ## Description
    Returns an array which contains all possible indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`.

    ## Arguments
    `dimension` (`int`): The dimension of the space.
    `num_grid` (`int`): The number of grids in each dimension.
    `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensor.

    ## Returns
    `torch.LongTensor`: A 1D tensor containing all possible index values.
    """
    return torch.arange((num_grid ** 2) * dimension + 1, dtype=dtype, device=device)


def freq_index_pair_tensor(
    dimension:     int,
    num_grid:      int,
    diagonal_only: bool                   = False,
    dtype:         torch.dtype            = torch.long,
    device:        Optional[torch.device] = None,
) -> torch.LongTensor:
    """Return an array containing all possible pairs of indices.

    ## Description
    Returns an array which contains all possible pairs of indices, i.e., all possible values of `|l+m|_2^2` and `|l-m|_2^2`.

    ## Arguments
    `dimension` (`int`): The dimension of the space.
    `num_grid` (`int`): The number of grids in each dimension.
    `diagonal_only` (`bool`, default: `False`): Determines whether only diagonal pairs are returned.
    `dtype` (`torch.dtype`, default: `torch.long`): The data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): The device of the output tensor.

    ## Returns
    `torch.LongTensor`: The tensor containing index pairs.
    """
    arr: torch.LongTensor = freq_index_tensor(dimension, num_grid, dtype=dtype, device=device)
    if diagonal_only:
        arr = arr.reshape(-1, 1)
        _zeros: torch.Tensor = torch.zeros_like(arr, dtype=dtype, device=device)
        return torch.stack((arr, _zeros), dim=-1)
    else:
        return torch.stack(torch.meshgrid(arr, arr, indexing='ij'), dim=-1)


def freq_slices_low(n_modes: Sequence[int]) -> tuple[tuple[slice, slice], ...]:
    """Return slices corresponding to low-frequency modes.

    ## Description
    Returns a tuple of slice pairs corresponding to the positive and negative low-frequency modes for each dimension.

    ## Arguments
    `n_modes` (`Sequence[int]`): The number of modes along each dimension to be preserved.

    ## Returns
    `tuple[tuple[slice, slice], ...]`: Slices for the low-frequency modes.
    """
    kernel_slices: list[tuple[slice, slice]] = []
    for n in n_modes:
        n_front: int = (n + 1) // 2
        n_rear:  int = n // 2
        kernel_slices.append((slice(None, n_front), slice(-n_rear, None)))
    return tuple(kernel_slices)


def fft_prod_slices(
    ndim:    int,
    dim:     Sequence[int],
    n_modes: Sequence[int],
) -> product:
    """Return the product of slices for FFT operations on low-frequency modes.

    ## Description
    Returns the Cartesian product of slices for the FFT operations acted on low-frequency modes.

    ## Arguments
    `ndim` (`int`): The number of dimensions of the input tensor.
    `dim` (`Sequence[int]`): The dimensions on which the FFT operations are acted.
    `n_modes` (`Sequence[int]`): The number of modes in each dimension to be preserved.

    ## Returns
    `product`: A product of slices, where each slice corresponds to the low-frequency modes in the specified dimensions.
    """
    _slices: list[Sequence[slice]] = [tuple([slice(None)])] * ndim
    for d, n in zip(dim, n_modes):
        n_front: int = (n + 1) // 2
        n_rear:  int = n // 2
        _slices[d] = tuple((slice(None, n_front), slice(-n_rear, None)))
    return product(*_slices)


def fft_compression(
    X_fft:            torch.Tensor,
    dim:              Sequence[int],
    compression_size: Sequence[int],
) -> torch.Tensor:
    """Compress the FFT of a signal to a lower-resolution FFT.

    ## Description
    Compress the FFT of a low-resolution signal to the corresponding high-resolutional FFT.
    The complex tensor is returned where only the low-frequency modes of `X_fft` are preserved.

    ## Arguments
    `X_fft` (`torch.Tensor`): The input tensor in the frequency domain.
    `dim` (`Sequence[int]`): The dimensions on which the compression is applied.
    `compression_size` (`Sequence[int]`): The sizes of the compressed dimensions.

    ## Returns
    `torch.Tensor`: The compressed complex tensor.
    """
    if len(dim) != len(compression_size):
        raise ValueError(
            f"The length of 'dim' ({len(dim)}) and 'compression_size' ({len(compression_size)}) should be the same."
        )
    ndim: int = X_fft.ndim
    dim = tuple((d % ndim for d in dim))
    for d, size_c in zip(dim, compression_size):
        size_x: int = X_fft.size(d)
        if size_x < size_c:
            raise ValueError(
                f"At dimension {d}, the input tensor is of size {size_x}, while the compression size is {size_c}."
            )

    newshape: list[int] = list(X_fft.shape)
    for d, s in zip(dim, compression_size):
        newshape[d] = s

    Y_fft: torch.Tensor = torch.zeros(list(newshape), dtype=X_fft.dtype, device=X_fft.device)
    for sl in fft_prod_slices(ndim, dim, compression_size):
        Y_fft[*sl] = X_fft[*sl]
    return Y_fft


def fft_expansion(
    X_fft:          torch.Tensor,
    dim:            Sequence[int],
    expansion_size: Sequence[int],
) -> torch.Tensor:
    """Expand the FFT of a signal to a higher-resolution FFT.

    ## Description
    Expand the FFT of a low-resolution signal to the corresponding high-resolutional FFT.
    The low-frequency modes of `X_fft` are expanded to high-frequency modes by zero-padding.

    ## Arguments
    `X_fft` (`torch.Tensor`): The input tensor in the frequency domain.
    `dim` (`Sequence[int]`): The dimensions on which the expansion is applied.
    `expansion_size` (`Sequence[int]`): The sizes of the expanded dimensions.

    ## Returns
    `torch.Tensor`: The expanded complex tensor.
    """
    if len(dim) != len(expansion_size):
        raise ValueError(
            f"The length of 'dim' ({len(dim)}) and 'expansion_size' ({len(expansion_size)}) should be the same."
        )
    ndim: int = X_fft.ndim
    dim = tuple((d % ndim for d in dim))
    for d, size_e in zip(dim, expansion_size):
        size_x: int = X_fft.size(d)
        if size_x > size_e:
            raise ValueError(
                f"At dimension {d}, the input tensor is of size {size_x}, while the expansion size is {size_e}."
            )

    newshape: list[int] = list(X_fft.shape)
    for d, s in zip(dim, expansion_size):
        newshape[d] = s
    n_modes: tuple[int, ...] = tuple([X_fft.size(d) for d in dim])

    Y_fft: torch.Tensor = torch.zeros(list(newshape), dtype=X_fft.dtype, device=X_fft.device)
    for sl in fft_prod_slices(ndim, dim, n_modes):
        Y_fft[*sl] = X_fft[*sl]
    return Y_fft


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()