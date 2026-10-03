"""PyTorch-based FFT operations package.

This package provides various FFT operations, including convolutions and utilities for handling frequency modes.
"""
from deep_numerical.fft.freq      import (
    fft_compression,
    fft_expansion,
    fft_index,
    fft_prod_slices,
    freq_index_pair_tensor,
    freq_index_tensor,
    freq_pair_tensor,
    freq_slices_low,
    freq_tensor,
)
from deep_numerical.fft.operation import (
    circular_convolution,
    convolve_freqs,
    convolve_signals,
    linear_convolution,
)


__all__: list[str] = [
    'fft_index',
    'freq_tensor',
    'freq_pair_tensor',
    'freq_index_tensor',
    'freq_index_pair_tensor',
    'freq_slices_low',
    'fft_prod_slices',
    'fft_compression',
    'fft_expansion',
    'convolve_signals',
    'convolve_freqs',
    'linear_convolution',
    'circular_convolution',
]


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()