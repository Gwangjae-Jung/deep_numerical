from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    .general    import  MLP
from    ..utils     import  EINSUM_STRING, get_activation


##################################################
##################################################
__all__: list[str] = ["SeparableSpectralConv", "SeparableFourierLayer"]


##################################################
##################################################
# Layers for the Separable Fourier Neural Operator
class SeparableSpectralConv(nn.Module):
    """## Separable spectral convolutional layer
    
    The convolution of two *real-valued* functions as the multiplication of their Fourier transforms.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int]   = None,
            rank:           Optional[int]   = 2,
        ) -> Self:
        """The initializer of the class `SeparableSpectralConv`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved.
                * The length of `n_modes` should be less than or equal to 26.
            `in_channels` (`int`):
                * The number of the input features.
            `out_channels` (`int`, default: `None`):
                * The number of the output features.
                * If `None`, then `out_channels` is set `in_channels`.
            `rank` (`int`, default: `2`):
                * The rank of the kernel.
                * This argument is not used in this class, but it is kept for compatibility with the `FactorizedSpectralConv` class.
        """
        if out_channels is None:
            out_channels = in_channels
        self.__check_arguments(n_modes, in_channels, out_channels, rank)
            
        super().__init__()
        dim_domain  = len(n_modes)
        
        self.__n_modes      = n_modes
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__rank         = rank
        self.__fft_dim      = tuple(range(-(1+dim_domain), -1, 1))
        self.__fft_norm     = "forward"
        self.__config_kernel()
        
        self.kernel_components = nn.ParameterList(
            [
                torch.rand(
                    size    = (*_shape, out_channels, in_channels, rank),
                    dtype   = torch.cfloat,
                )
                for _shape in self.__kernel_shape
            ]
        )
        
        return
    
    
    def __check_arguments(
            self,
            n_modes:        tuple[int],
            in_channels:    int,
            out_channels:   int,
            rank:           int,
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
        if not isinstance(rank, int) or rank<1:
            raise ValueError(f"The value of 'rank' is {rank}, which is not a positive integer.")
        return
    
    
    def __config_kernel(self) -> None:
        # NOTE (rfftn): Modification for the last dimension is required
        # Set `self.__kernel_shape`
        self.__kernel_shape: list[tuple[int]] = []
        for k in range(self.dim_domain):
            s = self.__n_modes[k] if k<self.dim_domain-1 else (self.__n_modes[k])//2+1
            _shape = [1 for _ in range(self.dim_domain)]
            _shape[k] = s
            self.__kernel_shape.append(tuple(_shape))
        
        # Set einsum commands
        self.__kernel_einsum_cmd:   str = \
            ','.join(["...ijr" for _ in self.__n_modes]) + "->...ij"
        self.__einsum_cmd:          str = \
            f"...ij,b...j->b...i"
                
        # Set `self.__kernel_slices`
        self.__kernel_slices:  list[list[slice]] = []
        for k in range(self.dim_domain):
            n_modes_front  = (self.__n_modes[k]+1) // 2
            n_modes_back   = self.__n_modes[k] - n_modes_front
            _slices: list[slice]
            if k<self.dim_domain-1:
                _slices = [slice(None, n_modes_front, None), slice(-n_modes_back, None, None)]
            else:
                _slices = [slice(None, n_modes_front, None)]
            self.__kernel_slices.append(_slices)
        
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
        kernel = torch.einsum(self.__kernel_einsum_cmd, *self.kernel_components)
        for kernel_slice in product(*self.__kernel_slices):
            Y_rfftn[:, *kernel_slice] = \
                torch.einsum(self.__einsum_cmd, kernel[*kernel_slice], X_rfftn[:, *kernel_slice])
        
        # Inverse fast Fourier transform (real version)
        return torch.fft.irfftn(Y_rfftn, dim=self.__fft_dim, s=X_spatial_shape, norm=self.__fft_norm)
        
    
    def __repr__(self) -> str:
        return f"SeparableSpectralConv(n_modes={self.__n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels}, rank={self.__rank})"

    
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
    def rank(self) -> int:
        return self.__rank
    @property
    def dim_domain(self) -> int:
        return len(self.__n_modes)


class SeparableFourierLayer(nn.Module):
    """## Separable Fourier layer
    
    The separable Fourier layer is a combination of a linear layer and a spectral convolutional layer.
    Note that the activation function is not included in this layer.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int] = None,
            rank:           Optional[int] = 2,
            
            activation_name:    str     = "relu",
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
            `rank` (`int`, default: `2`):
                * The rank of the kernel.
                * This argument is not used in this class, but it is kept for compatibility with the `FactorizedSpectralConv` class.
            `activation_name` (`str`, default: `"relu"`):
                * The name of the activation function to be used in the MLP.
                * The activation function is not included in this layer.
            `activation_kwargs` (`dict`, default: `{}`):
                * The keyword arguments to be passed to the activation function.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels    = in_channels
        
        # Define the subnetworks
        self.spectral   = SeparableSpectralConv(n_modes, in_channels, out_channels, rank)
        self.mlp        = MLP([out_channels, 2*out_channels, out_channels], activation_name=activation_name, activation_kwargs=activation_kwargs)
        return
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the `FourierLayer` class.
        
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor to be transformed.
                * The shape of `X` is expected to be `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the MLP and the spectral convolution, with the residual connection.
        """
        _spectral   = self.spectral.forward(X)
        return X + self.mlp.forward(_spectral)
    
    
    def __repr__(self) -> str:
        return f"SeparableFourierLayer(n_modes={self.n_modes}, in_channels={self.in_channels}, out_channels={self.out_channels}, rank={self.rank})"
    
    
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
    def rank(self) -> int:
        return self.spectral.rank
    @property
    def dim_domain(self) -> int:
        return self.spectral.dim_domain


##################################################
##################################################
# End of file