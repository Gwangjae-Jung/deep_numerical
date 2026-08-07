from    typing      import  Union, Optional, List, override
import  torch
from    torch       import  nn
from    itertools   import  product
from    deep_numerical                      import  EINSUM_STRING
from    deep_numerical.fft                  import  freq_tensor
from    deep_numerical.neural               import  get_activation
from    deep_numerical.neural.layer.general import  MLP


__all__ = ["RadialSpectralConv", "RadialFourierLayer"]


##################################################
##################################################
def _generate_mask(dimension: int, max_freq: int) -> torch.Tensor:
    mask = freq_tensor(dimension, 1+2*max_freq, True)[..., :1+max_freq, :]
    mask = mask.pow(2).sum(-1)[..., None, None] # Two more dimensions to match the dimension of the kernel
    return torch.where(mask<=max_freq**2, 1, 0)


class RadialSpectralConv(nn.Module):
    """## Radial spectral convolutional layer - The spectral convolutional layer with radial symmetry

    The convolution of two *real-valued* functions as the multiplication of their Fourier transforms.
    In particular, this layer provides the Fourier transform of a radially symmetric function, which is invariant under orthogonal transforms on the real Euclidean spaces, enforcing the equivariance under the orthogonal group.
    """
    def __init__(
            self,
            dim_domain:     int,
            max_freq:       int,
            in_channels:    int,
            out_channels:   Optional[int]           = None,
            dtype:          Optional[torch.dtype]   = None,
        ) -> None:
        """The initializer of the class `RadialSpectralConv`

        Arguments:
            `dim_domain` (`int`):
                The dimension of the domain.
            `max_freq` (`int`):
                The maximum frequency index to be preserved.
            `in_channels` (`int`):
                The number of the input features.
            `out_channels` (`int`, default: `None`):
                The number of the output features.
                If `None`, then `out_channels` is set `in_channels`.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
                If `None`, then the default complex data type is used.
        """
        if out_channels is None:
            out_channels = in_channels
        self.__check_arguments(dim_domain, max_freq, in_channels, out_channels)
            
        super().__init__()
        domain_str  = EINSUM_STRING[:dim_domain]
        
        self.__dim_domain   = dim_domain
        self.__max_freq     = max_freq
        self.__in_channels  = in_channels
        self.__out_channels = out_channels
        self.__einsum_cmd   = f"{domain_str}ij,b{domain_str}j->b{domain_str}i"
        self.__fft_dim      = tuple(range(-(1+dim_domain), -1, 1))
        self.__fft_norm     = "forward"
        self.__config_kernel()
        
        """The kernel tensor should be reconstructed from this base tensor."""
        len_base = 1+dim_domain*(max_freq**2)
        self.__kernel_base = nn.Parameter(torch.rand(size=(len_base, out_channels, in_channels), dtype=self.__dtype))
        self.register_buffer("kernel_mask", _generate_mask(dim_domain, max_freq), False)
        self.__frozen_kernel: Optional[torch.Tensor] = None
        
        return
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def max_freq(self) -> int:
        return self.__max_freq
    @property
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def out_channels(self) -> int:
        return self.__out_channels
    @property
    def dtype(self) -> torch.dtype:
        """The data type of the kernel tensor, which is complex."""
        return self.__kernel_base.dtype
    @property
    def dtype_real(self) -> torch.dtype:
        """The data type of the real-valued tensor."""
        if self.dtype==torch.cfloat:
            return torch.float
        elif self.dtype==torch.cdouble:
            return torch.double
        else:
            raise TypeError(f"The dtype {self.dtype} is not supported.")
    @property
    def dtype_complex(self) -> torch.dtype:
        """Alias of `RadialSpectralConv.dtype`."""
        return self.dtype

    
    @override
    def train(self, mode: bool=True) -> None:
        """Customized method `train()`."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        if mode:
            self.__frozen_kernel = None
        else:
            self.__frozen_kernel = self.construct_kernel().data
        return self
    @override
    def eval(self) -> None:
        """Customized method `eval()`."""
        return self.train(False)
    
    
    def __check_arguments(
            self,
            dim_domain:     int,
            max_freq:       int,
            in_channels:    int,
            out_channels:   int,
        ) -> None:
        if not isinstance(dim_domain, int) or dim_domain<=0:
            raise ValueError(f"'dim_domain' should be a positive integer not greater than {len(EINSUM_STRING)}, but got {dim_domain}.")
        if dim_domain>len(EINSUM_STRING):
            raise NotImplementedError(
                ' '.join([
                    f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_STRING)}.",
                    f"The passed 'dim_domain' is {dim_domain}."
                ])
            )
        if max_freq<0:
            raise ValueError(f"'max_freq' should be given as a positive real number, but got {max_freq}.")
        for ch, name in zip((in_channels, out_channels), ('in_channels', 'out_channels')):
            if not isinstance(ch, int) or ch<1:
                raise ValueError(f"The value of '{name}' is {ch}, which is not a positive integer.")
        return
    
    
    def __config_kernel(self) -> None:
        # Set `self.__construction_indices`
        n_modes_front   = self.__max_freq + 1
        n_modes_back    = self.__max_freq
        self.__construction_indices = freq_tensor(
            self.__dim_domain,
            1 + 2*self.__max_freq,
            keepdim = True,
            dtype   = torch.long,
        )[..., :n_modes_front, :]   # Relevant to `rfftn`
        self.__construction_indices = self.__construction_indices.pow(2).sum(-1)
        
        # Set `self.__kernel_slices`
        self.__kernel_slices:  List[List[slice]] = []
        for _ in range(self.__dim_domain-1):
            self.__kernel_slices.append([
                slice(None, n_modes_front, None),
                slice(-n_modes_back, None, None)
            ])
        self.__kernel_slices.append([slice(None, n_modes_front, None)])
        
        return
    
    
    def construct_kernel(self, as_complex: bool=True) -> Union[torch.Tensor, torch.nn.Parameter]:
        """Constructs the kernel tensor from the base tensor."""
        # Reshape the base tensor to the kernel shape
        if self.training or self.__frozen_kernel is None:
            mask: torch.Tensor = getattr(self, "kernel_mask")
            kernel = self.__kernel_base[self.__construction_indices] * mask
            if as_complex:
                kernel = kernel.type(self.dtype_complex)
            return kernel
        else:
            return self.__frozen_kernel
    
    
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
        X_spatial_shape = X.shape[-(1 + self.__dim_domain): -1]
        
        # Fast Fourier transform (real version)
        X_rfftn: torch.Tensor   = torch.fft.rfftn(X, dim=self.__fft_dim, norm=self.__fft_norm)
        
        # Instantiate a tensor which will be filled with the reduced FFT
        Y_rfftn = torch.zeros(size=(*X_rfftn.shape[:-1], self.__out_channels), dtype=self.dtype_complex, device=X.device)

        # Linear transform on the Fourier modes
        kernel = self.construct_kernel()
        for kernel_slice in product(*self.__kernel_slices):
            Y_rfftn[:, *kernel_slice] = \
                torch.einsum(self.__einsum_cmd, kernel[*kernel_slice], X_rfftn[:, *kernel_slice])
        
        # Inverse fast Fourier transform (real version)
        return torch.fft.irfftn(Y_rfftn, dim=self.__fft_dim, s=X_spatial_shape, norm=self.__fft_norm)
        
    
    def __repr__(self) -> str:
        return f"RadialSpectralConv(dim_domain={self.__dim_domain}, max_freq={self.__max_freq}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"




class RadialFourierLayer(nn.Module):
    """## Radial Fourier layer
    
    The Fourier layer is a combination of a linear layer and a spectral convolutional layer.
    Note that the activation function is not included in this layer.
    """
    def __init__(
            self,
            dim_domain:         int,
            max_freq:           int,
            in_channels:        int,
            out_channels:       Optional[int] = None,
            weighted_residual:  bool = True,
            bias:               bool = True,
            
            activate:           bool    = True,
            activation_name:    str     = 'relu',
            activation_kwargs:  dict    = {},
            
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """The initializer of the class `FourierLayer`
        
        Arguments:
            `dim_domain` (`int`):
                The dimension of the domain.
            `max_freq` (`int`):
                The maximum frequency index to be preserved.
            `in_channels` (`int`):
                The number of the input features.
            `out_channels` (`int`, default: `None`):
                The number of the output features.  If `None`, then `out_channels` is set `in_channels`.
            `weighted_residual` (`bool`, default: `True`):
                Whether to use a weighted residual connection.
                If `True`, a linear layer is used in the skip connection.
                If `False`, the skip connection is a simple addition, so it should be ensured that `in_channels==out_channels`. Instead, a 2-layer MLP will be used after the spectral convolution, and the activation function is not applied after the residual connection.
            `bias` (`bool`, default: `True`):
                Whether to use bias terms in the linear layers.
            `activate` (`bool`, default: `True`):
                Whether to apply the activation function.
            `activation_name` (`str`, default: `'relu'`):
                The name of the activation function to be used.
            `activation_kwargs` (`dict`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
                If `None`, then the default real data type is used.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
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
        self.spectral   = RadialSpectralConv(dim_domain, max_freq, in_channels, out_channels, dtype=dtype)
        
        # Define the activation function
        self.activation = get_activation(activation_name, activation_kwargs) if activate else nn.Identity()
        return
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass of the `FourierLayer` class.
        
        Arguments:
            `X` (`torch.Tensor`): The input tensor of shape of `X` is expected to be `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` are the spatial dimensions, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The transformed tensor after applying the linear and spectral convolutions.
        """
        _linear     = self.linear.forward(X)
        _spectral   = self.spectral.forward(X)
        out = _linear + self.mlp.forward(_spectral)
        out = self.activation.forward(out)
        return out
    
    
    def __repr__(self) -> str:
        return f"FourierLayer(dim_domain={self.dim_domain}, max_freq={self.max_freq}, in_channels={self.__in_channels}, out_channels={self.__out_channels})"
    
    
    @property
    def dim_domain(self) -> int:
        return self.spectral.dim_domain
    @property
    def max_freq(self) -> int:
        return self.spectral.max_freq
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
# End of file