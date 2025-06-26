from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    ..utils     import  EINSUM_STRING, get_activation, freq_tensor
from    .general    import  MLP


##################################################
##################################################
__all__: list[str] = ["RadialSpectralConv", "RadialFourierLayer"]


##################################################
##################################################
# Layers for the Fourier Neural Operator
class RadialSpectralConv(nn.Module):
    """## Radial spectral convolutional layer - The spectral convolutional layer with radial symmetry

    The convolution of two *real-valued* functions as the multiplication of their Fourier transforms.
    In particular, this layer provides the Fourier transform of a radially symmetric function, which is invariant under orthogonal transforms on the real Euclidean spaces, enforcing the equivariance under the orthogonal group.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int]   = None,
            dtype:          torch.dtype     = torch.float,
        ) -> Self:
        """The initializer of the class `RadialSpectralConv`

        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved.
                * The length of `n_modes` should be less than or equal to the length of `EINSUM_STRING`.
                * *Required*: `n_modes` should be a tuple of single integers.
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
        
        self.__dtype: torch.dtype = dtype
        
        len_base = 1 + sum(( (n//2)**2 for n in n_modes ))
        self.__kernel_base = nn.Parameter(
            torch.rand(
                size = (len_base, out_channels, in_channels),
                dtype = self.dtype_complex,
            )
        )
        """The kernel tensor should be reconstructed from this base tensor."""
        
        return
    
    
    @property
    def dtype(self) -> torch.dtype:
        """The data type of the kernel tensor."""
        return self.__dtype
    @property
    def dtype_real(self) -> torch.dtype:
        """The data type of the real-valued tensor."""
        return self.__dtype
    @property
    def dtype_complex(self) -> torch.dtype:
        """The data type of the complex-valued tensor."""
        return torch.cfloat if self.__dtype is torch.float else torch.cdouble
    
    
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
        if len(set(n_modes)) > 1:
            raise ValueError(
                "The spectral convolutional layer is not equivariant under the orthogonal group when the Fourier modes are not isotropic.",
            )
        return
    
    
    def __config_kernel(self) -> None:
        # NOTE (fftn): Radian spectral convolution is implemented using the complex FFT.
        # Set `self.__construction_indices`
        self.__construction_indices = freq_tensor(
            self.dim_domain,
            self.__n_modes[0],
            keepdim = True,
            dtype   = torch.long,
        )
        self.__construction_indices = self.__construction_indices.pow(2).sum(-1)
        
        # # Set `self.__kernel_shape`
        # self.__kernel_shape: list[int]   = list(self.__n_modes)
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  list[list[slice]] = []
        for k in range(self.dim_domain):
            n_modes_front  = (self.__n_modes[k]+1) // 2
            n_modes_back   = self.__n_modes[k] - n_modes_front
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_back, None, None)
            ])
        
        return
    
    
    def construct_kernel(self) -> torch.Tensor:
        """Constructs the kernel tensor from the base tensor."""
        # Reshape the base tensor to the kernel shape
        return self.__kernel_base[self.__construction_indices]
    
    
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
        X_rfftn: torch.Tensor   = torch.fft.fftn(X, dim=self.__fft_dim, norm=self.__fft_norm)
        
        # Instantiate a tensor which will be filled with the reduced FFT
        Y_rfftn = torch.zeros(size=(*X_rfftn.shape[:-1], self.__out_channels), dtype=self.dtype_complex, device=X.device)

        # Linear transform on the Fourier modes
        kernel = self.construct_kernel()
        for kernel_slice in product(*self.__kernel_slices):
            Y_rfftn[:, *kernel_slice] = \
                torch.einsum(self.__einsum_cmd, kernel[*kernel_slice], X_rfftn[:, *kernel_slice])
        
        # Inverse fast Fourier transform (real version)
        return torch.real(torch.fft.ifftn(Y_rfftn, dim=self.__fft_dim, s=X_spatial_shape, norm=self.__fft_norm))
        
    
    def __repr__(self) -> str:
        return f"RadialSpectralConv(n_modes={self.__n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"

    
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


class RadialFourierLayer(nn.Module):
    """## Radial Fourier layer
    
    The Fourier layer is a combination of a linear layer and a spectral convolutional layer.
    Note that the activation function is not included in this layer.
    """
    def __init__(
            self,
            n_modes:            Sequence[int],
            in_channels:        int,
            out_channels:       Optional[int] = None,
            weighted_residual:  bool = True,
            
            activate:           bool    = True,
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
                * If `False`, the skip connection is a simple addition. Instead, a 2-layer MLP will be used after the spectral convolution, and the activation function is not applied after the residual connection.
                
            `activate` (`bool`, default: `True`):
                * If `True`, the activation function is applied.
                * If `False`, the activation function is not applied.
            `activation_name` (`str`, default: `'relu'`):
                * The name of the activation function to be used.
            `activation_kwargs` (`dict`, default: `{}`):
                * The keyword arguments for the activation function.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels    = in_channels
        if weighted_residual == False:
            activate = False
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__activate     = activate
        
        # Define the subnetworks
        self.linear = nn.Linear(in_channels, out_channels) if weighted_residual else \
            nn.Identity()
        self.mlp    = nn.Identity() if weighted_residual else \
            MLP(
                [out_channels, 2*out_channels, out_channels],
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
        self.spectral   = RadialSpectralConv(n_modes, in_channels, out_channels)
        
        # Define the activation function
        self.activation = get_activation(activation_name, activation_kwargs) if activate else nn.Identity()
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
        out: torch.Tensor = _linear + self.mlp.forward(_spectral)
        if self.__activate:
            out = self.activation.forward(out)
        return out
    
    
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