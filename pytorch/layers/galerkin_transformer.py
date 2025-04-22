from    typing              import  *
from    typing_extensions   import  Self

from    math        import  prod
import  torch
from    torch       import  nn

from    ..utils     import  EINSUM_STRING
from    .general    import  MLP




##################################################
##################################################
__all__ = [
    "GalerkinTypeSelfAttention",
    "GalerkinTypeCrossAttention",
    "GalerkinTypeEncoderBlockSelfAttention",
    "GalerkinTypeEncoderBlockCrossAttention",
]


##################################################
##################################################
class GalerkinTypeSelfAttention(nn.Module):
    """## Galerkin-type self-attention
    ### Self-attention with the key-value inner product ahead
    """
    def __init__(
            self,
            dim_domain:         int,
            hidden_channels:    int,
            n_heads:            int = 1,
        ) -> Self:
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
        _size   = (hidden_channels, n_heads, hidden_channels//n_heads)    # (C, H, D), where D=C/H
        _scale  = (hidden_channels ** 2) / n_heads
        # Feature maps
        self.sa_query   = nn.Parameter(torch.randn(size=_size, dtype=torch.float) / _scale)
        self.sa_key     = nn.Parameter(torch.randn(size=_size, dtype=torch.float) / _scale)
        self.sa_value   = nn.Parameter(torch.randn(size=_size, dtype=torch.float) / _scale)
        # Layer normalization
        self.layernorm_key   = nn.LayerNorm((n_heads, hidden_channels//n_heads))
        self.layernorm_value = nn.LayerNorm((n_heads, hidden_channels//n_heads))
        
        return
    
    
    @property
    def einsum_domain(self) -> str:
        return self.__EINSUM_DOMAIN
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        return f"GalerkinTypeSelfAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"




class GalerkinTypeCrossAttention(nn.Module):
    """## Galerkin-type cross-attention
    ### Self-attention with the key-value inner product ahead
    """
    def __init__(
            self,
            dim_domain:         int,        # Query
            hidden_channels:    int,        # Key and value
            n_heads:            int = 1,    # Common
        ) -> Self:
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
        _size_U = (hidden_channels, n_heads, hidden_channels // n_heads)    # (C, H, C/H)
        _size_X = (dim_domain, n_heads, hidden_channels // n_heads)         # (D, H, C/H)
        _scale  = (hidden_channels ** 2) / n_heads
        # Feature maps
        self.ca_query   = nn.Parameter(torch.randn(size = _size_X, dtype = torch.float) / _scale)
        self.ca_key     = nn.Parameter(torch.randn(size = _size_U, dtype = torch.float) / _scale)
        self.ca_value   = nn.Parameter(torch.randn(size = _size_U, dtype = torch.float) / _scale)
        # Layer normalization
        self.layernorm_key   = nn.LayerNorm((n_heads, hidden_channels // n_heads))
        self.layernorm_value = nn.LayerNorm((n_heads, hidden_channels // n_heads))
        
        return
    
    
    @property
    def einsum_domain(self) -> str:
        return self.__EINSUM_DOMAIN
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    
    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        ### Arguments
        @ `U` (`torch.Tensor`)
            * `U` is the embedding of the input function.
            * `U` has the shape `(B, *__domain__. C)`.
            * `U` is input to the key map and the value map.
        
        @ `X` (`torch.Tensor`)
            * `X` is the 3-tensor saving the coordinates of the query points.
            * `X` has the shape `(B, size(__domain__), dim(__domain__))`.
            * `X` is input to the query map.
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
        U_kv = torch.einsum(f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde", [U_key,   U_value ])
        U_sa = torch.einsum(f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh", [X_query, U_kv    ])
        U_sa = (U_sa / num_points).reshape(U.shape)
        
        return U_sa

    
    def __repr__(self) -> str:
        return f"GalerkinTypeCrossAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"


##################################################
##################################################
class GalerkinTypeEncoderBlockSelfAttention(nn.Module):
    """## Galerkin-type self-attention
    ### Self-attention with the key-value inner product ahead
    """
    def __init__(
            self,
            dim_domain:             int,
            hidden_channels:        int,
            n_heads:                int = 1,
            mlp_hidden_channels:    Sequence[int] = None,
            mlp_activation_name:    str                 = "relu",
            mlp_activation_kwargs:  dict[str, object]   = {},
        ) -> Self:
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"The number of the heads in the self attention should divide the number of the hidden features. "
                f"('hidden_channels': {hidden_channels}, 'n_heads': {n_heads})"
            )
        super().__init__()
        
        if mlp_hidden_channels is None:
            mlp_hidden_channels = [hidden_channels * 2]
        
        self.__dim_domain       = dim_domain
        self.__hidden_channels  = hidden_channels
        self.__n_heads          = n_heads
        
        self.sa  = GalerkinTypeSelfAttention(dim_domain, hidden_channels, n_heads)
        self.mlp = MLP(
            [hidden_channels] + mlp_hidden_channels + [hidden_channels],
            activation_name     = mlp_activation_name,
            activation_kwargs   = mlp_activation_kwargs
        )
        
        return

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.sa.forward(X)
        X = X + self.mlp.forward(X)
        return X
        

    def __repr__(self) -> str:
        return f"GalerkinTypeEncoderBlockSelfAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"




class GalerkinTypeEncoderBlockCrossAttention(nn.Module):
    """## Galerkin-type cross-attention
    ### Self-attention with the key-value inner product ahead
    """
    def __init__(
            self,
            dim_input_domain:       int,
            dim_query_domain:       int,
            hidden_channels:        int,
            n_heads:                int = 1,
            mlp_hidden_channels:    Sequence[int] = None,
            mlp_activation_name:    str                 = "relu",
            mlp_activation_kwargs:  dict[str, object]   = {},
        ) -> Self:
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"The number of the heads in the self attention should divide the number of the hidden features. "
                f"('hidden_channels': {hidden_channels}, 'n_heads': {n_heads})"
            )
        super().__init__()
        
        if mlp_hidden_channels is None:
            mlp_hidden_channels = [hidden_channels * 2]
        
        self.__dim_input_domain = dim_input_domain
        self.__dim_query_domain = dim_query_domain
        self.__hidden_channels  = hidden_channels
        self.__n_heads          = n_heads
        
        self.ca  = GalerkinTypeCrossAttention(dim_input_domain, hidden_channels, n_heads)
        self.mlp = MLP(
            [hidden_channels] + mlp_hidden_channels + [hidden_channels],
            activation_name     = mlp_activation_name,
            activation_kwargs   = mlp_activation_kwargs
        )
        
        return

    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        ### Arguments
        * `U` (`torch.Tensor`)
            * `U` is the embedding of the input function.
            * `U` has the shape `(B, *__domain__, C)`.
            * `U` is input to the key map and the value map.
        
        * `X` (`torch.Tensor`)
            * `X` is the 3-tensor saving the coordinates of the query points.
            * `X` has the shape `(B, size(__domain__), dim(__domain__))`.
            * `X` is input to the query map.
        """
        U = U + self.ca.forward(U, X)
        U = U + self.mlp.forward(U)
        return U
        

    def __repr__(self) -> str:
        return f"GalerkinTypeEncoderBlockCrossAttention({self.__dim_input_domain=}, {self.__dim_query_domain=}, {self.__hidden_channels=}, {self.__n_heads=})"


##################################################
##################################################
# End of file