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
            mlp_channels:   Optional[Sequence[int]] = None,
        ) -> Self:
        if out_channels is None:
            out_channels = in_channels
        if mlp_channels is None:
            mlp_channels = [in_channels, in_channels+out_channels, out_channels]
        
        self.__check_arguments(n_modes, in_channels, out_channels, mlp_channels)
        
        super().__init__()
        n_modes     = tuple(n_modes)
        dim_domain  = len(n_modes)
        
        self.__n_modes      = n_modes
        self.__dim_domain   = dim_domain
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        # domain_str  = EINSUM_STRING[:dim_domain]
        # self.__einsum_cmd   = f"{domain_str}ij,b{domain_str}j->b{domain_str}i"
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
        self.mlp = MLP(mlp_channels)
        
        return
    
    
    def __check_arguments(
            self,
            n_modes:        tuple[int],
            in_channels:    int,
            out_channels:   int,
            mlp_channels:   Sequence[int],
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
        for ch in mlp_channels:
            if not isinstance(ch, int) or ch<1:
                raise ValueError(f"The value of 'mlp_channels' is {mlp_channels}, which is not a positive integer.")
        return
    
    
    def __config_kernel(self) -> None:
        # NOTE (rfftn): Modification for the last dimension is required
        # Set `self.__kernel_shape`
        self.__kernel_shape: list[int]   = list(self.__n_modes)
        self.__kernel_shape[-1]          = self.__n_modes[-1]//2 + 1
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  list[list[slice]] = []
        for k in range(self.__dim_domain-1):
            n_modes_front  = (self.__n_modes[k]+1) // 2
            n_modes_back   = self.__n_modes[k] - n_modes_front
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_back, None, None)
            ])
        self.__kernel_slices.append([ slice(None, self.__kernel_shape[-1], None) ])
        
        # Set `self.__einsum_strings`
        self.__einsum_strings:  list[str]   = []
        domain_string = EINSUM_STRING[:self.__dim_domain]
        for d in range(self.__dim_domain):
            domain_string__d = domain_string
            domain_string__d[d] = 'r'
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
            for _kernel_slice in self.__kernel_slices[d]:
                _range =  [...,]
                _range += [self.__kernel_slices[d]]
                _range += [slice(None)] * (self.__dim_domain-d)
                Y[*_range] = torch.einsum(
                    self.__einsum_strings[d],
                    X_rfft_d[*_range], self.kernel_list[d][_kernel_slice]
                )
            
            # Add to the output
            summand = torch.fft.irfft(Y, dim=fft_dim, norm=self.__fft_dim, n=X_size_d)
            out += summand
        
        out = self.mlp.forward(out)
        return out
    # def forward(self, X: torch.Tensor) -> torch.Tensor:
    #     out = torch.zeros_like(X, dtype=torch.float, device=X.device)
    #     for d in range(self.__dim_domain):
    #         # Discrete Fourier transform for each dimension
    #         fft_dim     = (X.ndim-1-self.dim_domain)+d
    #         X_rfft_d    = torch.fft.rfft(X, dim = fft_dim, norm = fft_norm)
            
    #         # Instantiate a tensor which will be filled with a reduced DFT
    #         X_rfft_d_shape  = list(X_rfft_d.shape)
    #         X_spatial_shape = list(X.shape)[-self.dim_domain:]
    #         Y = torch.zeros(size = X_rfft_d_shape, dtype = torch.cfloat, device = X.device)
            
    #         # Get the range of the low-frequency parts of the discrete Fourier transform
    #         _range =  [slice(None), slice(None)]
    #         _range += [slice(None)] * d
    #         _range += [slice(self.n_modes[d])]
    #         _range += [slice(None)] * (self.dim_domain - d - 1)
            
    #         # Do the affine transform
    #         Y[_range] = self.factorized_spectral_affine(X_rfft_d[_range], dim = d)
            
    #         # Add to the output
    #         summand = torch.fft.irfft(Y, dim = fft_dim, norm = fft_norm, n = X_spatial_shape[d])
    #         out += summand
    

##################################################
##################################################
class FactorizedFourierLayer(nn.Module):
    """## Factorized Fourier layer
    ### Kernel integration via dimensionwise FFT with a skip connection
    
    -----
    Given a batch of uniformly discretized functions, this class computes the spectral convolution of the input with a skip connection.
    
    -----
    ## Remark
    
    1. The input tensor should be of shape `(batch_size, hidden_channels, *shape_of_domain)`.
    2. The skip connection is implemented by adding the result of the spectral convolution with a linear transform of the input tensor. The linear transform in the skip connection is shared by all points of the domain.
    3. As for the spectral convolution, see the documentation of the class `SpectralConv`.
    """
    
    def __init__(
                    self,
                    n_modes:            Sequence[int],
                    hidden_channels:    int,
                    expansion_factor:   float,
                    activation:         str
        ) -> None:
        
        super().__init__()
        
        self.hidden_channels    = hidden_channels
        self.n_modes            = n_modes
        self.dim_domain         = len(n_modes)
        
        self.activation = getattr(nn, TORCH_ACTIVATION_DICT[activation])()
        
        conv = getattr(nn, f"Conv{self.dim_domain}d", None)
        if conv is None:
            raise NotImplementedError(
                f"The PyTorch library provides the convolutional layers up to dimenion 3, but the domain is {self.dim_domain}-dimensional."
            )
        
        self.spectral = nn.Sequential(
            FactorizedSpectralConv(
                n_modes         = n_modes,
                hidden_channels = hidden_channels
            ),
            conv(
                in_channels     = hidden_channels,
                out_channels    = int(expansion_factor * hidden_channels),
                kernel_size     = 1
            ),
            self.activation,
            conv(
                in_channels     = int(expansion_factor * hidden_channels),
                out_channels    = hidden_channels,
                kernel_size     = 1
            ),
            self.activation
        )
                
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X + self.spectral(X))
    
        


##################################################
##################################################
