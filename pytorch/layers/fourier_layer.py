from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    ..utils         import  EINSUM_STRING


##################################################
##################################################
__all__: list[str] = ["SpectralConv", "FourierLayer"]


##################################################
##################################################
class FastFourierTransformLayer(nn.Module):
    """## A layer for fast Fourier transform (forward and inverse)"""
    def __init__(self, dim: Sequence[int], norm: str='forward',) -> Self:
        super().__init__()
        self.__dim  = tuple(dim)
        self.__norm = norm
        return
    
    
    @property
    def dim(self) -> tuple[int]:
        return self.__dim
    @property
    def dimension(self) -> int:
        return len(self.__dim)
    @property
    def fft_norm(self) -> str:
        return self.__norm
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(X, dim=self.__dim, norm=self.__norm)
        
        
    def inverse(
            self,
            X_fft:      torch.Tensor,
            shape:      Optional[Sequence[int]] = None,
            is_real:    bool                    = True,
        ) -> torch.Tensor:
        inv: torch.Tensor = torch.fft.ifftn(X_fft, dim=self.__dim, s=shape, norm=self.__norm)
        return inv.real if is_real else inv
    
    
    def __repr__(self) -> str:
        return f"FastFourierTransformLayer(dim={self.__dim}, norm={self.__norm})"



##################################################
##################################################
# Layers for Fourier Neural Operator
class SpectralConv(nn.Module):
    """## Spectral convolutional layer
    ### Physical convolution as a linear transform in the Fourier space
    
    -----
    ### Description
    In this class, the spectral convolution, which computes the (periodic) convolution of a function with a kernel, is implemented.
    
    -----
    ### Implementation
    Let the input tensor `X` of shape `(batch_size, *shape_of_domain, hidden_channels)` is given.
    
    1. The (multidimensional) real fast fourier transform (rFFT) of `X` is computed.
    2. Among all rFFT modes, only the rFFT modes of low frequencies are preserved. For these rFFT modes, a mode-wise defined linear transform is done.
    3. After the mode-wise linear transform, the inverse rFFT (IrFFT) is computed and returned.
    
    -----
    ### Note
    1. As this class conducts the rFFT, in the last dimension of the Fourier transform, the Fourier modes of negative frequencies are deprecated. This should not be done if one conducts the complex FFT.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int]   = None,
        ) -> Self:
        """## The initializer of the class `SpectralConv`
        
        -----
        ### Some member variables
        1. `__list_size__` (`list[int]`)
            The list of the size of the tensor which behaves as the integrating kernel, i.e., the weight in the Fourier space.
        
        2. `__list_range__` (`list[slice]`)
            The list of ranges to be used in the forward propagation.
        """
        if (len(n_modes) > len(EINSUM_STRING)):
            raise NotImplementedError(
                f"The spectral convolutional layer is not supported "
                f"for dimensions greater than {len(EINSUM_STRING)}."
            )
        n_modes: tuple[int] = tuple(n_modes)
        if out_channels is None:
            out_channels = in_channels
        self.__check_arguments
            
        super().__init__()
        dim_domain  = len(n_modes)
        domain_str  = EINSUM_STRING[:dim_domain]
        
        self.__n_modes      = n_modes
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__einsum_cmd   = f"{domain_str}ij,b{domain_str}j->b{domain_str}i"
        self.__fft_dim      = tuple(range(-(1+dim_domain), -1, 1))
        self.__fft_norm     = "forward"
        self.__config_kernel()
        
        # The linear transform is not shared for the Fourier modes
        self.kernel = nn.Parameter(
            torch.rand(size=(*self.__kernel_shape, out_channels, in_channels), dtype=torch.cfloat)
        )
        
        return
    
    
    def __check_arguments(n_modes: tuple[int], in_channels: int, out_channels: int) -> None:
        if len(n_modes)>len(EINSUM_STRING):
            raise NotImplementedError(f"'SpectralConv' supports the spectral convolution for dimension less than or equal to {len(EINSUM_STRING)}.")
        for d, n in enumerate(n_modes):
            if not isinstance(n, int) or n<1:
                raise ValueError(f"The value of 'n_modes[{d}]' is {n}, which is not a positive integer.")
        for ch, name in zip((in_channels, out_channels), ('in_channels', 'out_channels')):
            if not isinstance(ch, int) or ch<1:
                raise ValueError(f"The value of '{name}' is {ch}, which is not a positive integer.")
        return
    
    
    def __config_kernel(self) -> None:
        # NOTE (rfftn): Modification for the last dimension is required
        # Set `self.__kernel_shape`
        self.__kernel_shape: list[int]   = list(self.__n_modes)
        self.__kernel_shape[-1]          = self.__n_modes[-1]//2 + 1
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  list[list[slice]] = []
        for k in range(self.dim_domain-1):
            n_modes_front  = (self.__n_modes[k]+1) // 2
            n_modes_rear   = self.__n_modes[k] - n_modes_front
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_rear, None, None)
            ])
        self.__kernel_slices.append([ slice(None, self.__kernel_shape[-1], None) ])
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Backup the spatial shape of `X` (Required to restore the shape of the convolution)
        X_spatial_shape = X.shape[-(1 + self.dim_domain): -1]
        
        # Fast Fourier transform (real version)
        X_rfftn: torch.Tensor   = torch.fft.rfftn(X, dim=self.__fft_dim, norm=self.__fft_norm)
        
        # Instantiate a tensor which will be filled with the reduced FFT
        Y_rfftn = torch.zeros(size=(*X_rfftn.shape[:-1], self.__out_channels), dtype=torch.cfloat, device=X.device)
        
        # Linear transform on the Fourier modes
        for kernel_slice in product(*self.__kernel_slices):
            print(Y_rfftn.shape, self.kernel[*kernel_slice].shape, X_rfftn[:, *kernel_slice].shape)
            Y_rfftn[:, *kernel_slice] = \
                torch.einsum(self.__einsum_cmd, self.kernel[*kernel_slice], X_rfftn[:, *kernel_slice])
        
        # Inverse fast Fourier transform (real version)
        return torch.fft.irfftn(Y_rfftn, dim=self.__fft_dim, s=X_spatial_shape, norm=self.__fft_norm)
        
    
    def __repr__(self) -> str:
        return f"SpectralConv(n_modes={self.__n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"

    
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    @property
    def dim_domain(self) -> int:
        return len(self.__n_modes)


class FourierLayer(nn.Module):
    """## Fourier layer
    ### Kernel integration using the Fast Fourier Transform (FFT)
    
    -----
    ### Description
    This class generates an object which computes the pointwise linear transform and the FFT of uniformly discretized functions.
    
    -----
    ### Remark
    1. The input tensor should be of shape `(B, s_1, ..., s_d, C)`.
    2. For more information about the spectral convolution, see the docstring of the class `SpectralConv`.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int] = None,
        ) -> Self:
        """## The initializer of the class `FourierLayer`
        
        -----
        ### Arguments
        * `n_modes` (`Sequence[int]`)
            * The maximum degree of the Fourier modes to be preserved.
        
        * `in_channels` (`int`)
            * The number of the input features.
        
        * `out_channels` (`int`, default: `None`)
            * The number of the output features.
            * If `None`, then `out_channels` is set `in_channels`.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        
        # Define the subnetworks
        self.linear     = nn.Linear(in_channels, out_channels)
        self.spectral   = SpectralConv(n_modes, in_channels, out_channels)
        return
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        ### Note
        1. This class assumes that the input tensor `X` has the shape `(B, s_1, ..., s_d, C)`.
        """
        _linear     = self.linear.forward(X)
        _spectral   = self.spectral.forward(X)
        return _linear + _spectral
    
    
    def __repr__(self) -> str:
        return f"FourierLayer(n_modes={self.n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"
    
    
    @property
    def n_modes(self) -> int:
        return self.spectral.n_modes


##################################################
##################################################
# Layers for Adaptive Fourier Neural Operator


##################################################
##################################################
# End of file