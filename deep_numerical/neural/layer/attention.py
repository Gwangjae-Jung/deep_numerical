from    typing      import  Sequence, Dict, Optional
from    math        import  prod
import  torch
from    torch       import  nn
from    deep_numerical  import  EINSUM_STRING
from    deep_numerical.neural   import  Activations, get_activation
from    deep_numerical.neural.layer.general     import  MLP

try:
    from    .vector_self_attention  import  VectorSelfAttention, _HAS_TORCH_GEOMETRIC
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH_GEOMETRIC = False

__all__ = ["LinearSelfAttention", "LinearCrossAttention", "ModifiedMLP", "HyperLinearSelfAttention"]
if _HAS_TORCH_GEOMETRIC:
    __all__.append("VectorSelfAttention")



class LinearSelfAttention(nn.Module):
    """Linear self-attention
    
    -----
    ### Description
    This class aims to compute the self-attention of given features from two distinct sources.
    The modifications from the original attention layer are listed below:
    
    * The softmax is removed, and the key-value multiplication is computed ahead of the multiplication with the query. It reduces the quadratic complexity to the linear complexity.
    * The scaling is done by the inverse of the length of the sequence, rather than its square root.
    """
    def __init__(
            self,
            dim_domain:         int,
            hidden_channels:    int,
            n_heads:            int = 1,
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """The initializer of `LinearSelfAttention`.
        
        Arguments:
            `dim_domain` (`int`): The dimension of the domain in which the points exist. `dim_domain` is used to compute the einsum command.
            `hidden_channels` (`int`): The number of the hidden channels. `hidden_channels` is used to compute the einsum command.
            `n_heads` (`int`, default: `1`): The number of the heads in the attention layer.
            `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the parameters. If `None`, the default data type of `torch` will be used.
        """
        # Check if the number of the hidden channels is divisible by the number of the heads
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"The number of the heads in the self attention should divide the number of the hidden features. "
                f"('hidden_channels': {hidden_channels}, 'n_heads': {n_heads})"
            )
        
        # Initialization begins
        super().__init__()
        
        # Save some variables for representation and computation
        self.__dim_domain       = dim_domain
        self.__hidden_channels  = hidden_channels
        self.__n_heads          = n_heads
        self.__EINSUM_DOMAIN    = EINSUM_STRING[:self.__dim_domain]
        self.__EINSUM_COMMAND   = f"b{self.__EINSUM_DOMAIN}c,chd->b{self.__EINSUM_DOMAIN}hd"
        
        # Variables for attention
        _size   = (hidden_channels, n_heads, hidden_channels//n_heads)    # (C, H, C/H)
        _scale  = (hidden_channels**2) / n_heads
        _dtype  = dtype if dtype is not None else torch.get_default_dtype()
        config_params = {'size': _size, 'dtype': _dtype}
        # Feature maps
        self.sa_query   = nn.Parameter(torch.randn(**config_params) / _scale)
        self.sa_key     = nn.Parameter(torch.randn(**config_params) / _scale)
        self.sa_value   = nn.Parameter(torch.randn(**config_params) / _scale)
        # Layer normalization
        self.layernorm_key   = nn.LayerNorm((n_heads, hidden_channels//n_heads), dtype=_dtype)
        self.layernorm_value = nn.LayerNorm((n_heads, hidden_channels//n_heads), dtype=_dtype)
        
        return None
    
    
    @property
    def einsum_domain(self) -> str:
        return self.__EINSUM_DOMAIN
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward propagation of `LinearSelfAttention`.
        
        Arguments:
            `X` (`torch.Tensor`): The input tensor.
        
        Returns:
            `torch.Tensor`: The linear self-attention of the input tensor `X`.
        """
        # Map to the query/key/value spaces
        X_query = torch.einsum(self.einsum_command, [X, self.sa_query])
        X_key   = torch.einsum(self.einsum_command, [X, self.sa_key])
        X_value = torch.einsum(self.einsum_command, [X, self.sa_value])
        
        # Layer normalization
        X_key   = self.layernorm_key.forward(X_key)
        X_value = self.layernorm_value.forward(X_value)
        
        # Compute the Galerkin-type self-attention
        num_points = prod(X.shape[1: 1 + self.__dim_domain])
        X_kv = torch.einsum(f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde", [X_key,   X_value ])
        X_sa = torch.einsum(f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh", [X_query, X_kv    ])
        X_sa = (X_sa / num_points).reshape(X.shape)
        
        return X_sa

    
    def __repr__(self) -> str:
        return f"LinearSelfAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"


class LinearCrossAttention(nn.Module):
    """Linear cross-attention
    
    -----
    ### Description
    This class aims to compute the cross-attention of given features from two distinct sources.
    The modifications from the original attention layer are listed below:
    
    * The softmax is removed, and the key-value multiplication is computed ahead of the multiplication with the query. It reduces the quadratic complexity to the linear complexity.
    * The scaling is done by the inverse of the length of the sequence, rather than its square root.
    """
    def __init__(
            self,
            dim_domain:         int,        # Query
            hidden_channels:    int,        # Key and value
            n_heads:            int = 1,    # Common
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """The initializer of `LinearCrossAttention`.
        
        Arguments:
            `dim_domain` (`int`): The dimension of the domain in which the points exist. `dim_domain` is used to compute the einsum command.
            `hidden_channels` (`int`): The number of the hidden channels. `hidden_channels` is used to compute the einsum command.
            `n_heads` (`int`, default: `1`): The number of the heads in the attention layer.
        """
        # Check if the number of the hidden channels is divisible by the number of the heads
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"The number of the heads in the self attention should divide the number of the hidden features. "
                f"('hidden_channels': {hidden_channels}, 'n_heads': {n_heads})"
            )
        
        # Initialization begins
        super().__init__()
        
        # Save some variables for representation and computation
        self.__dim_domain       = dim_domain
        self.__hidden_channels  = hidden_channels
        self.__n_heads          = n_heads
        self.__EINSUM_DOMAIN    = EINSUM_STRING[:self.__dim_domain]
        self.__EINSUM_COMMAND   = f"b{self.__EINSUM_DOMAIN}c,chd->b{self.__EINSUM_DOMAIN}hd"
        
        # Variables for attention
        _size_U = (hidden_channels, n_heads, hidden_channels//n_heads)    # (C, H, C/H)
        _size_X = (dim_domain, n_heads, hidden_channels//n_heads)         # (D, H, C/H)
        _scale  = (hidden_channels ** 2) / n_heads
        dtype   = dtype if dtype is not None else torch.get_default_dtype()
        # Feature maps
        self.ca_query   = nn.Parameter(torch.randn(size=_size_X, dtype=dtype) / _scale)
        self.ca_key     = nn.Parameter(torch.randn(size=_size_U, dtype=dtype) / _scale)
        self.ca_value   = nn.Parameter(torch.randn(size=_size_U, dtype=dtype) / _scale)
        # Layer normalization
        self.layernorm_key   = nn.LayerNorm((n_heads, hidden_channels//n_heads), dtype=dtype)
        self.layernorm_value = nn.LayerNorm((n_heads, hidden_channels//n_heads), dtype=dtype)
        
        return None
    
    
    @property
    def einsum_domain(self) -> str:
        return self.__EINSUM_DOMAIN
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    
    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """The forward propagation of `LinearCrossAttention`.
        
        Arguments:
            `U` (`torch.Tensor`):
                * `U` is the embedding of the input function.
                * `U` has the shape `(B, *shape_of_domain, C)`.
                * `U` is input to the key map and the value map.
        
            `X` (`torch.Tensor`):
                * `X` is a tensor saving the coordinates of the query points.
                * `X` has the shape `(B, *shape_of_domain, dim_domain)`.
                * `X` is input to the query map.
        
        Returns:
            `torch.Tensor`:
                * A `torch.Tensor` object of shape `(B, *shape_of_domain, C)`.
        """
        # Map to the query/key/value spaces
        X_query = torch.einsum(self.einsum_command, [X, self.ca_query])
        U_key   = torch.einsum(self.einsum_command, [U, self.ca_key])
        U_value = torch.einsum(self.einsum_command, [U, self.ca_value])
        
        # Layer normalization
        U_key   = self.layernorm_key.forward(U_key)
        U_value = self.layernorm_value.forward(U_value)
        
        # Compute the Galerkin-type self-attention
        num_points = prod(U.shape[1: 1 + self.__dim_domain])
        U_kv = torch.einsum(
            f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde",
            U_key, U_value,
        )
        U_ca = torch.einsum(
            f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh",
            X_query, U_kv,
        )
        U_ca = (U_ca/num_points).reshape(U.shape)
        
        return U_ca

    
    def __repr__(self) -> str:
        return f"LinearCrossAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"






class ModifiedMLP(nn.Module):
    """Modified MLP
    
    -----
    ### Reference
    https://epubs.siam.org/doi/epdf/10.1137/20M1318043
    """
    def __init__(
            self,
            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            n_layers:           int = 4,
            
            activation_name:    Activations         = "relu",
            activation_kwargs:  Dict[str, object]   = {},
        ) -> None:
        super().__init__()
        self.__in_channels     = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels    = out_channels
        self.__n_layers         = n_layers
        self.network_basis1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            get_activation(activation_name, **activation_kwargs),
        )
        self.network_basis2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            get_activation(activation_name, **activation_kwargs),
        )
        self.network_trans = nn.ModuleList(
            [
                nn.Linear(in_channels, hidden_channels),
                get_activation(activation_name, **activation_kwargs),
            ] + [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    get_activation(activation_name, **activation_kwargs),
                ) for _ in range(n_layers-1)
            ]
        )
        self.network_coeff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    get_activation(activation_name, **activation_kwargs),
                ) for _ in range(n_layers)
            ]
        )
        self.network_project = nn.Linear(hidden_channels, out_channels)
        return None
    
    
    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        return self.__in_channels

    @property
    def hidden_channels(self) -> int:
        """Number of hidden channels."""
        return self.__hidden_channels

    @property
    def out_channels(self) -> int:
        """Number of output channels."""
        return self.__out_channels

    @property
    def n_layers(self) -> int:
        """Number of hidden layers."""
        return self.__n_layers
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes forward pass of ModifiedMLP.

        ## Description
        Applies input-basis gating across hidden layers using transform and coefficient sub-networks.

        ## Arguments
        `X` (`torch.Tensor`): Input tensor of shape `(..., in_channels)`.

        ## Returns
        `torch.Tensor`: Output tensor of shape `(..., out_channels)`.
        """
        U = self.network_basis1.forward(X)
        V = self.network_basis2.forward(X)
        for cnt in range(self.__n_layers):
            X       = self.network_trans[cnt].forward(X)
            coeff   = self.network_coeff[cnt].forward(X)
            X = (1-coeff)*U + coeff*V
        out = self.network_project.forward(X)
        return out
    


class HyperLinearSelfAttention(nn.Module):
    """Linear self-attention with hyper-networks
    
    -----
    ### Description
    This class aims to compute the self-attention of given features from two distinct sources.
    The modifications from the original attention layer are listed below:
    
    * The softmax is removed, and the key-value multiplication is computed ahead of the multiplication with the query. It reduces the quadratic complexity to the linear complexity.
    * The scaling is done by the inverse of the length of the sequence, rather than its square root.
    """
    def __init__(
            self,
            dim_domain:         int,
            hidden_channels:    int,
            hyper_channels:     Sequence[int],
            n_heads:            int = 1,
        ) -> None:
        # Check if the number of the hidden channels is divisible by the number of the heads
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"The number of the heads in the self attention should divide the number of the hidden features. "
                f"('hidden_channels': {hidden_channels}, 'n_heads': {n_heads})"
            )
        
        # Initialization begins
        super().__init__()
        
        # Save some variables for representation and computation
        self.__dim_domain       = dim_domain
        self.__hidden_channels  = hidden_channels
        self.__n_heads          = n_heads
        self.__EINSUM_DOMAIN    = EINSUM_STRING[:self.__dim_domain]
        self.__EINSUM_COMMAND   = f"b{self.__EINSUM_DOMAIN}c,bchd->b{self.__EINSUM_DOMAIN}hd"
        
        # Variables for hyper-attention
        self.__qkv_shape = (-1, hidden_channels, n_heads, hidden_channels//n_heads)
        qkv_size = hidden_channels**2
        self.sa_query   = MLP((*hyper_channels, qkv_size))
        self.sa_key     = MLP((*hyper_channels, qkv_size))
        self.sa_value   = MLP((*hyper_channels, qkv_size))
        # Layer normalization
        self.layernorm_key   = nn.LayerNorm((n_heads, hidden_channels//n_heads))
        self.layernorm_value = nn.LayerNorm((n_heads, hidden_channels//n_heads))
        
        return None
    
    
    @property
    def einsum_domain(self) -> str:
        """Spatial domain einsum subscript."""
        return self.__EINSUM_DOMAIN

    @property
    def einsum_command(self) -> str:
        """Einsum command string for multi-head linear projection."""
        return self.__EINSUM_COMMAND
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Computes hypernetwork linear self-attention forward pass.

        ## Description
        Generates query, key, and value projection weights from hyper-parameters `p` via MLPs,
        and computes linear self-attention over `X`.

        ## Arguments
        `X` (`torch.Tensor`): Input tensor of shape `(batch, *domain, channels)`.
        `p` (`torch.Tensor`): Hypernetwork parameter tensor conditioning the projections.

        ## Returns
        `torch.Tensor`: Output tensor of shape `(batch, *domain, channels)`.
        """
        # Map to the query/key/value spaces
        w_query = self.sa_query.forward(p).reshape(*self.__qkv_shape)
        w_key   = self.sa_key.forward(p).reshape(*self.__qkv_shape)
        w_value = self.sa_value.forward(p).reshape(*self.__qkv_shape)
        X_query = torch.einsum(self.einsum_command, [X, w_query])
        X_key   = torch.einsum(self.einsum_command, [X, w_key])
        X_value = torch.einsum(self.einsum_command, [X, w_value])
        
        # Layer normalization
        X_key   = self.layernorm_key.forward(X_key)
        X_value = self.layernorm_key.forward(X_value)
        
        # Compute the Galerkin-type self-attention
        num_points = prod(X.shape[1: 1 + self.__dim_domain])
        X_kv = torch.einsum(f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde", [X_key,   X_value ])
        X_sa = torch.einsum(f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh", [X_query, X_kv])
        X_sa = (X_sa/num_points).reshape(X.shape)
        
        return X_sa

    
    def __repr__(self) -> str:
        return f"LinearSelfAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
