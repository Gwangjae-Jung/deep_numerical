from    typing      import  Optional, Sequence
import  torch


__all__ = [
    # FFT operations
    'convolve_signals',
    'convolve_freqs',
    'linear_convolution',
    'circular_convolution',
]

FFT_NORM:   str = 'forward'


##################################################
##################################################
# FFT operations
## Convolutions using FFT
def convolve_signals(
        x1:     torch.Tensor,
        x2:     torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Computes the convolution of two input signals using the fast Fourier transform.
    
    Arguments:
        `x1` (`torch.Tensor`):
            The first signal.
        `x2` (`torch.Tensor`):
            The second signal.
        `dim` (`Optional[Sequence[int]]`, default: `None`):
            The sequence of dimensions to be convolved.
            If `None`, then `x1` and `x2` are convolved throughout the entire dimensions.

    ### Note
    As the operation held on the frequency domain is just the pointwise multiplication, the output does not suffer from aliasing.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input tensors are not of the same shape.\n"
            f"* x1.shape: {list(x1.shape)}\n"
            f"* x2.shape: {list(x2.shape)}\n"
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
    """Computes the convolution of two input FFTs using the fast Fourier transform.
    
    Arguments:
        `x1_fft` (`torch.Tensor`):
            The first FFT.
        `x2_fft` (`torch.Tensor`):
            The second FFT.
        `dim` (`Optional[Sequence[int]]`, default: `None`):
            The sequence of dimensions to be convolved.
            If `None`, then `x1` and `x2` are convolved throughout the entire dimensions.
    
    ### Note
    The output of this function is valid only for the *finite* signals.
    Hence, the output may not be used to recover the multiplication of the original signals, unless the sampling frequency is not less than the Nyquist frequency.
    """
    if x1_fft.shape != x2_fft.shape:
        raise ValueError(
            f"Two input tensors are not of the same shape.\n"
            f"* x1_fft.shape: {list(x1_fft.shape)}\n"
            f"* x2_fft.shape: {list(x2_fft.shape)}\n"
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
    """Returns the circular convolution of two input signals.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input tensors are not of the same shape.\n"
            f"* x1.shape: {list(x1.shape)}\n"
            f"* x2.shape: {list(x2.shape)}\n"
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
    """Returns the linear convolution of two input signals.
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"Two input arrays are not of the same shape.\n"
            f"* x1.shape: {list(x1.shape)}\n"
            f"* x2.shape: {list(x2.shape)}\n"
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