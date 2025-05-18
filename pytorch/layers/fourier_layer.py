from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    ..utils         import  EINSUM_STRING, get_activation
from    ..layers        import  MLP


##################################################
##################################################
__all__: list[str] = ["FFTLayer", "SpectralConv", "FourierLayer"]


##################################################
##################################################
class FFTLayer(nn.Module):
    """## A layer for the fast Fourier transform (forward and inverse)"""
    def __init__(self, dim: Sequence[int], norm: str='forward') -> Self:
        """The initializer of the class `FFTLayer`
        
        Arguments:
            `dim` (`Sequence[int]`): The dimensions of the input tensor to be transformed.
            `norm` (`str`, default: `'forward'`): The normalization mode for the FFT."""
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
        """Computes the FFT of the input tensor `X` along the specified dimensions.
        
        Arguments:
            `X` (`torch.Tensor`): The input tensor to be transformed.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the FFT.
        """
        return torch.fft.fftn(X, s=X.shape[*self.__dim], dim=self.__dim, norm=self.__norm)
        
        
    def inverse(
            self,
            X_fft:      torch.Tensor,
            shape:      Optional[Sequence[int]] = None,
            is_real:    bool                    = True,
        ) -> torch.Tensor:
        """Computes the inverse FFT of the input tensor `X_fft` along the specified dimensions.
        
        Arguments:
            `X_fft` (`torch.Tensor`): The input tensor to be transformed.
            `shape` (`Optional[Sequence[int]]`, default: `None`):
                * The shape of the input tensor.
                * If `None`, then the shape is set to the shape of `X_fft`.
            `is_real` (`bool`, default: `True`):
                * If `True`, the output tensor is returned as a real-valued tensor.
                * If `False`, the output tensor is returned as a complex-valued tensor.
        """
        inv: torch.Tensor = torch.fft.ifftn(X_fft, s=shape, dim=self.__dim, norm=self.__norm)
        return inv.real if is_real else inv
    
    
    def __repr__(self) -> str:
        return f"FFTLayer(dim={self.__dim}, norm={self.__norm})"



##################################################
##################################################
# Layers for the Fourier Neural Operator
class SpectralConv(nn.Module):
    """## Spectral convolutional layer
    
    The convolution of two *real-valued* functions as the multiplication of their Fourier transforms.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int]   = None,
        ) -> Self:
        """The initializer of the class `SpectralConv`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved.
                * The length of `n_modes` should be less than or equal to the length of `EINSUM_STRING`.
            `in_channels` (`int`):
                * The number of the input features.
            `out_channels` (`int`, default: `None`):
                * The number of the output features.
                * If `None`, then `out_channels` is set `in_channels`.
        """
        if out_channels is None:
            out_channels = in_channels
        self.__check_arguments(n_modes, in_channels, out_channels)
            
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
            torch.rand(
                size    = (*self.__kernel_shape, out_channels, in_channels),
                dtype   = torch.cfloat,
            )
        )
        
        return
    
    
    def __check_arguments(
            self,
            n_modes:        tuple[int],
            in_channels:    int,
            out_channels:   int,
        ) -> None:
        if (len(n_modes) > len(EINSUM_STRING)):
            raise NotImplementedError(
                ' '.join([
                    f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_STRING)}.",
                    f"The passed 'n_modes' is {n_modes}, whose length is {len(n_modes)}."
                ])
            )
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
            n_modes_back   = self.__n_modes[k] - n_modes_front
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_back, None, None)
            ])
        self.__kernel_slices.append([ slice(None, self.__kernel_shape[-1], None) ])
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the spectral convolution of the input tensor `X`.
        
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor to be transformed.
                * The shape of `X` is expected to be `(B, s_1, ..., s_d, C)`, where `B` is the batch size, s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the spectral convolution.
        """
        # Backup the spatial shape of `X` (Required to restore the shape of the convolution)
        X_spatial_shape = X.shape[-(1 + self.dim_domain): -1]
        
        # Fast Fourier transform (real version)
        X_rfftn: torch.Tensor   = torch.fft.rfftn(X, dim=self.__fft_dim, norm=self.__fft_norm)
        
        # Instantiate a tensor which will be filled with the reduced FFT
        Y_rfftn = torch.zeros(size=(*X_rfftn.shape[:-1], self.__out_channels), dtype=torch.cfloat, device=X.device)
        
        # Linear transform on the Fourier modes
        for kernel_slice in product(*self.__kernel_slices):
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
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def out_channels(self) -> int:
        return self.__out_channels
    @property
    def dim_domain(self) -> int:
        return len(self.__n_modes)


class FourierLayer(nn.Module):
    """## Fourier layer
    
    The Fourier layer is a combination of a linear layer and a spectral convolutional layer.
    Note that the activation function is not included in this layer.
    """
    def __init__(
            self,
            n_modes:            Sequence[int],
            in_channels:        int,
            out_channels:       Optional[int] = None,
            weighted_residual:  bool = True,
            
            activation_name:    str     = 'relu',
            activation_kwargs:  dict    = {},
        ) -> Self:
        """The initializer of the class `FourierLayer`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved.
            `in_channels` (`int`):
                * The number of the input features.
            `out_channels` (`int`, default: `None`):
                * The number of the output features.
                * If `None`, then `out_channels` is set `in_channels`.
            `weighted_residual` (`bool`, default: `True`):
                * If `True`, a linear layer is used in the skip connection.
                * If `False`, the skip connection is a simple addition. Instead, a 2-layer MLP will be used after the spectral convolution.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels    = in_channels
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        
        # Define the subnetworks
        self.linear     = nn.Linear(in_channels, out_channels) if weighted_residual else nn.Identity()
        self.mlp        = nn.Identity() if weighted_residual else \
            MLP(
                [out_channels, 2*out_channels, out_channels],
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
        self.spectral   = SpectralConv(n_modes, in_channels, out_channels)
        return
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the `FourierLayer` class.
        
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor to be transformed.
                * The shape of `X` is expected to be `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the linear and spectral convolutions.
        """
        _linear     = self.linear.forward(X)
        _spectral   = self.spectral.forward(X)
        return _linear + self.mlp.forward(_spectral)
    
    
    def __repr__(self) -> str:
        return f"FourierLayer(n_modes={self.n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"
    
    
    @property
    def n_modes(self) -> int:
        return self.spectral.n_modes
    @property
    def in_channels(self) -> int:
        return self.spectral.in_channels
    @property
    def out_channels(self) -> int:
        return self.spectral.out_channels
    @property
    def dim_domain(self) -> int:
        return self.spectral.dim_domain


##################################################
##################################################
# End of file