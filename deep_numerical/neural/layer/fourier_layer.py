from    typing      import  Sequence, Optional, List, Tuple
import  torch
from    torch       import  nn
from    itertools   import  product
from    deep_numerical                      import  EINSUM_STRING
from    deep_numerical.utils                import  type_as_complex
from    deep_numerical.neural               import  get_activation
from    deep_numerical.neural.utils         import  Activations
from    deep_numerical.neural.layer.general import  MLP


__all__ = ["SpectralConv", "FourierLayer"]


##################################################
##################################################
class SpectralConv(nn.Module):
    """## Spectral convolutional layer
    
    The convolution of two *real-valued* functions as the multiplication of their Fourier transforms.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   Optional[int]           = None,
            dtype:          Optional[torch.dtype]   = None,
        ) -> None:
        """The initializer of the class `SpectralConv`.
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                The maximum degree of the Fourier modes to be preserved.
                The length of `n_modes` should be less than or equal to the length of `EINSUM_STRING`.
            `in_channels` (`int`):
                The number of the input features.
            `out_channels` (`Optional[int]`, default: `None`):
                The number of the output features.
                If `None`, then `out_channels` is set `in_channels`.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
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
        self.__dtype = type_as_complex(dtype)
        self.kernel = nn.Parameter(torch.rand(size=(*self.__kernel_shape, out_channels, in_channels), dtype=self.__dtype))
        
        return
    
    
    def __check_arguments(
            self,
            n_modes:        Tuple[int, ...],
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
        self.__kernel_shape: List[int]   = list(self.__n_modes)
        self.__kernel_shape[-1]          = self.__n_modes[-1]//2 + 1
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  List[List[slice]] = []
        for k in range(self.dim_domain-1):
            n_modes_front  = (self.__n_modes[k]+1) // 2
            n_modes_back   = self.__n_modes[k] - n_modes_front
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_back, None, None)
            ])
        self.__kernel_slices.append([ slice(None, self.__kernel_shape[-1], None) ])
        
        return None
    
    
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
        Y_rfftn = torch.zeros(size=(*X_rfftn.shape[:-1], self.__out_channels), dtype=self.__dtype, device=X.device)
        
        # Linear transform on the Fourier modes
        for kernel_slice in product(*self.__kernel_slices):
            Y_rfftn[:, *kernel_slice] = torch.einsum(self.__einsum_cmd, self.kernel[*kernel_slice], X_rfftn[:, *kernel_slice])
        
        # Inverse fast Fourier transform (real version)
        return torch.fft.irfftn(Y_rfftn, dim=self.__fft_dim, s=X_spatial_shape, norm=self.__fft_norm)
        
    
    def __repr__(self) -> str:
        return f"SpectralConv(n_modes={self.__n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"

    
    @property
    def dim_domain(self) -> int:
        return len(self.__n_modes)
    @property
    def n_modes(self) -> Tuple[int, ...]:
        return self.__n_modes
    @property
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def out_channels(self) -> int:
        return self.__out_channels
    @property
    def dtype(self) -> torch.dtype:
        """The data type of the kernel tensor, which is complex."""
        return self.__dtype
    @property
    def dtype_real(self) -> torch.dtype:
        """The data type of the real-valued tensor."""
        if self.__dtype is torch.cfloat:    return torch.float
        elif self.__dtype is torch.cdouble: return torch.double
        else:   raise TypeError(f"The dtype {self.__dtype} is not supported.")
    @property
    def dtype_complex(self) -> torch.dtype:
        """Alias of `SpectralConv.dtype`."""
        return self.__dtype


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
            bias:               bool = True,
            
            activate:           bool        = True,
            activation_name:    Activations = 'relu',
            activation_kwargs:  dict        = {},
            
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """The initializer of the class `FourierLayer`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                The maximum degree of the Fourier modes to be preserved.
            `in_channels` (`int`):
                The number of the input features.
            `out_channels` (`int`, default: `None`):
                The number of the output features.
                If `None`, then `out_channels` is set `in_channels`.
            `weighted_residual` (`bool`, default: `True`):
                Whether to use a weighted residual connection.
                If `True`, a linear layer is used in the skip connection.
                If `False`, the skip connection is a simple addition, so it should be ensured that `in_channels==out_channels`. Instead, a 2-layer MLP will be used after the spectral convolution, and the activation function is not applied after the residual connection.
            `bias` (`bool`, default: `True`):
                Whether to include a bias term in the linear layer.
            `activate` (`bool`, default: `True`):
                If `True`, the activation function is applied.
                If `False`, the activation function is not applied.
            `activation_name` (`str`, default: `'relu'`):
                The name of the activation function to be used.
            `activation_kwargs` (`dict`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels    = in_channels
        if weighted_residual==False:
            if in_channels!=out_channels:
                raise ValueError(
                    ' '.join([
                        "When 'weighted_residual' is set to False,",
                        f"'in_channels' ({in_channels}) should be equal to 'out_channels' ({out_channels})."
                    ])
                )
            activate = False
        if dtype is None:   dtype = torch.get_default_dtype()
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__dtype        = dtype
        
        # Define the subnetworks
        self.linear = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype) if weighted_residual else nn.Identity()
        self.mlp    = nn.Identity() if weighted_residual else \
            MLP(
                [out_channels, 2*out_channels, out_channels],
                bias                = bias,
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
                dtype               = dtype,
            )
        self.spectral   = SpectralConv(n_modes, in_channels, out_channels, dtype=dtype)
        
        # Define the activation function
        self.activation = get_activation(activation_name, activation_kwargs) if activate else nn.Identity()

        return None
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the `FourierLayer` class.
        
        Arguments:
            `X` (`torch.Tensor`):
                The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the linear and spectral convolutions.
        """
        _linear     = self.linear.forward(X)
        _spectral   = self.spectral.forward(X)
        out = _linear + self.mlp.forward(_spectral)
        out = self.activation.forward(out)
        return out
    
    
    def __repr__(self) -> str:
        return f"FourierLayer(n_modes={self.n_modes}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"
    
    
    @property
    def dim_domain(self) -> int:
        return self.spectral.dim_domain
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
    def dtype(self) -> torch.dtype:
        return self.__dtype


##################################################
##################################################
if __name__=="__main__":
    # For quick test
    torch.set_default_dtype(torch.double)
    layer_1 = FourierLayer([17, 17], 3, 5, weighted_residual=True,  activation_name='tanh', bias=False)
    layer_2 = FourierLayer([17, 17], 3, 3, weighted_residual=False, activation_name='tanh', bias=False)
    x = torch.zeros(size=(2, 64, 64, 3))
    layer_1.eval()
    layer_2.eval()
    with torch.inference_mode():
        y1 = layer_1.forward(x)
        y2 = layer_2.forward(x)
    print(y1.dtype, y2.dtype)
    print(y1.shape, y2.shape)
    print(y1.norm(p=torch.inf), y2.norm(p=torch.inf))


##################################################
##################################################
# End of file