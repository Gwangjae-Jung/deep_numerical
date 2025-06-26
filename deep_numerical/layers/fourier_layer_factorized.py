from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn

from    .general    import  MLP
from    ..utils     import  EINSUM_STRING, TORCH_ACTIVATION_DICT


##################################################
##################################################
__all__: list[str] = [
    "FactorizedFourierLayer",
    "FactorizedSpectralConv",
]


##################################################
##################################################
class FactorizedSpectralConv(nn.Module):
    """## Factorized spectral convolutional layer
    ### A separable version of `SpectralConv`.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int] = None,
        ) -> Self:
        if out_channels is None:
            out_channels = in_channels
        
        self.__check_arguments(n_modes, in_channels, out_channels)
        
        super().__init__()
        n_modes     = tuple(n_modes)
        dim_domain  = len(n_modes)
        
        self.__n_modes      = n_modes
        self.__dim_domain   = dim_domain
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__fft_dim      = tuple(range(-(1+dim_domain), -1, 1))
        self.__fft_norm     = "forward"
        self.__config_kernel()
        
        scale_linear = 1/(in_channels*out_channels)
        self.kernel_list = nn.ParameterList(
            [
                nn.Parameter(
                    scale_linear * torch.randn(
                        size    = (n_mode, out_channels, in_channels),
                        dtype   = torch.cfloat,
                    )
                )
                for n_mode in self.__kernel_shape
            ]
        )
        
        return
    
    
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
        return self.__dim_domain
    
    
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
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  list[slice] = []
        for k in range(self.__dim_domain):
            self.__kernel_slices.append(slice(None, self.__n_modes[k], None))
        
        # Set `self.__einsum_strings`
        self.__einsum_strings:  list[str]   = []
        domain_string = EINSUM_STRING[:self.__dim_domain]
        for d in range(self.__dim_domain):
            domain_string__d = list(domain_string)
            domain_string__d[d] = 'r'
            domain_string__d = ''.join(domain_string__d)
            self.__einsum_strings.append(
                f"rij, b{domain_string__d}j -> b{domain_string__d}i"
            )
        return

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the spectral convolutional layer.
        
        Arguments:
            `X` (`torch.Tensor`):
                The input tensor of shape `(batch_size, s_1, ..., s_d, in_channels)`.
        
        Returns:
            `torch.Tensor`:
                The output tensor of shape `(batch_size, s_1, ..., s_d, out_channels)`.
        """
        out = torch.zeros_like(X, dtype=torch.float, device=X.device)
        for d, fft_dim in enumerate(self.__fft_dim):
            # Discrete Fourier transform for each dimension
            X_rfft_d:   torch.Tensor = torch.fft.rfft(X, dim=fft_dim, norm=self.__fft_norm)
            
            # Instantiate a tensor which will be filled with a reduced DFT
            X_size_d = X.size(fft_dim)
            Y = torch.zeros(size=X_rfft_d.shape, dtype=X_rfft_d.dtype, device=X.device)
            
            # Conduct the linear transform in the Fourier space
            __kernel_slice = self.__kernel_slices[d]
            _range =  [...]
            _range += [__kernel_slice]
            _range += [slice(None)] * (self.__dim_domain-d)
            Y[*_range] = torch.einsum(
                self.__einsum_strings[d],
                self.kernel_list[d][__kernel_slice], X_rfft_d[*_range]
            )
            
            # Add to the output
            summand = torch.fft.irfft(Y, dim=fft_dim, norm=self.__fft_norm, n=X_size_d)
            out += summand

        return out
    

##################################################
##################################################
class FactorizedFourierLayer(nn.Module):
    """## Factorized Fourier layer
    
    The factorized Fourier layer is a combination of a linear layer and a spectral convolutional layer.
    Note that the activation function is not included in this layer.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int] = None,
            mlp_channels:   Optional[Sequence[int]] = None,
            activation_name:    str             = "relu",
            activation_kwargs:  Optional[dict]  = None,
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
            `mlp_channels` (`Sequence[int]`, default: `None`):
                * The number of the channels in the MLP.
                * If `None`, then `mlp_channels` is set to `[out_channels, out_channels+in_channels, in_channels]`.
            `activation_name` (`str`, default: `"relu"`):
                * The name of the activation function, which is applied after the spectral convolution.
            `activation_kwargs` (`dict`, default: `None`):
                * The keyword arguments for the activation function.
        """
        super().__init__()
        if out_channels is None:
            out_channels    = in_channels
        if mlp_channels is None:
            mlp_channels    = [out_channels, out_channels+in_channels, in_channels]
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        
        # Define the subnetworks
        self.spectral   = FactorizedSpectralConv(n_modes, in_channels, out_channels)
        self.mlp        = MLP(mlp_channels, activation_name=activation_name, activation_kwargs=activation_kwargs)

        return
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the `FactorizedFourierLayer` class.
        
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor to be transformed.
                * The shape of `X` is expected to be `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the linear and spectral convolutions.
        """
        _spectral   = self.spectral.forward(X)
        return X + self.mlp.forward(_spectral)
    
    
    def __repr__(self) -> str:
        return f"FactorizedFourierLayer(n_modes={self.n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"
    
    
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
