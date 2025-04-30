from    typing                  import  *
from    typing_extensions       import  Self

from    math        import  prod

import  torch
from    torch       import  nn
from    torch_geometric.nn      import  MessagePassing

from    ..utils     import  EINSUM_STRING
from    ..layers    import  MLP


##################################################
##################################################
__all__ = [
    "LinearSelfAttention",
    "LinearCrossAttention",
    
    "VectorSelfAttention",
    
    "HyperLinearSelfAttention",
]


##################################################
##################################################
class LinearSelfAttention(nn.Module):
    """## Linear self-attention
    
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
        _size   = (hidden_channels, n_heads, hidden_channels//n_heads)    # (C, H, C/H)
        _scale  = (hidden_channels**2) / n_heads
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
        X_value = self.layernorm_key.forward(X_value)
        
        # Compute the Galerkin-type self-attention
        num_points = prod(X.shape[1: 1 + self.__dim_domain])
        X_kv = torch.einsum(f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde", [X_key,   X_value ])
        X_sa = torch.einsum(f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh", [X_query, X_kv    ])
        X_sa = (X_sa / num_points).reshape(X.shape)
        
        return X_sa

    
    def __repr__(self) -> str:
        return f"LinearSelfAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"




class LinearCrossAttention(nn.Module):
    """## Linear cross-attention
    
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
        _size_U = (hidden_channels, n_heads, hidden_channels//n_heads)    # (C, H, C/H)
        _size_X = (dim_domain, n_heads, hidden_channels//n_heads)         # (D, H, C/H)
        _scale  = (hidden_channels ** 2) / n_heads
        # Feature maps
        self.ca_query   = nn.Parameter(torch.randn(size = _size_X, dtype = torch.float) / _scale)
        self.ca_key     = nn.Parameter(torch.randn(size = _size_U, dtype = torch.float) / _scale)
        self.ca_value   = nn.Parameter(torch.randn(size = _size_U, dtype = torch.float) / _scale)
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
    
    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        ### Arguments
        * `U` (`torch.Tensor`)
            * `U` is the embedding of the input function.
            * `U` has the shape `(B, *__domain__. C)`.
            * `U` is input to the key map and the value map.
        
        9 `X` (`torch.Tensor`)
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
        U_value = self.layernorm_key.forward(U_value)
        
        # Compute the Galerkin-type self-attention
        num_points = prod(U.shape[1: 1 + self.__dim_domain])
        U_kv = torch.einsum(f"b{self.einsum_domain}hd,b{self.einsum_domain}he->bhde", [U_key,   U_value ])
        U_sa = torch.einsum(f"b{self.einsum_domain}hd,bhde->b{self.einsum_domain}eh", [X_query, U_kv    ])
        U_sa = (U_sa / num_points).reshape(U.shape)
        
        return U_sa

    
    def __repr__(self) -> str:
        return f"LinearCrossAttention(dim_domain: {self.__dim_domain}, hidden_channels: {self.__hidden_channels}, n_heads: {self.__n_heads})"


##################################################
##################################################
class VectorSelfAttention(MessagePassing):
    """## Vector self-attention
    
    -----
    ### Description
    This class aims to compute the vector-valued self-attention of given features from two distinct sources.
    The modifications from the original attention layer are listed below:
    
    ### Remark
    1. The softmax can be removed by passing `use_softmax=False` to the initializer.
    2. The layer normalization will not be used.
    """
    def __init__(
            self,
            
            channels:       int,
            n_heads:        int,
            pos_encoder:    nn.Module,
            
            use_softmax:    bool = True,
            use_linear:     bool = True,
        ) -> Self:
        # Initialization begins
        super().__init__(aggr='sum')
        
        # Save some variables for representation and computation
        self.__channels         = channels
        self.__n_heads          = n_heads
        self.__pos_encoder      = pos_encoder
        self.__use_softmax      = use_softmax
        self.__use_linear       = use_linear
        self.__EINSUM_COMMAND   = f"nc,chd->nhd"
        
        # Variables for attention
        _size   = (channels, n_heads, channels//n_heads)    # (C, H, C/H)
        _scale  = channels ** 2
        # Feature maps
        self.sa_query   = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
        self.sa_key     = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
        self.sa_value   = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
        # Internal linear layer as a skip connection
        self.linear     = nn.Linear(channels, channels) if self.__use_linear else nn.Identity()
        
        return
    
    
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    @property
    def channels(self) -> int:
        return self.__channels
    @property
    def n_heads(self) -> int:
        return self.__n_heads
    @property
    def use_softmax(self) -> bool:
        return self.__use_softmax
    @property
    def use_linear(self) -> bool:
        return self.__use_linear
    
    
    def pos_encode(self, position: torch.Tensor) -> torch.Tensor:
        """
        Given a position tensor of shape `(N, k)`, this method computes the positional encoding of shape (N, H, C/H).
        Here,
        * `N` is the number of the nodes (points),
        * `k` is the dimension of the space in which the points exist,
        * `H` is the number of the heads,
        * `C` is the hidden dimension.
        """
        pos_enc: torch.Tensor
        pos_enc = self.__pos_encoder.forward(position)
        pos_enc = pos_enc.reshape(-1, self.n_heads, self.channels//self.n_heads)
        return pos_enc
    
    
    def forward(
            self,
            node_attr:  torch.Tensor,
            edge_index: torch.LongTensor,
            position:   torch.Tensor,
        ) -> torch.Tensor:
        """## The forward method of `VectorSelfAttention`
        
        -----
        ### Arguments
        
        @ `node_attr` (`torch.Tensor`)
            * A `torch.Tensor` object of shape `(N, D)`, where `N` is the number of the points and `D` is the number of the input channels. Note that there is not dimension for the batch, which is in accordance with the operation of `torch_geometric`.
        
        @ `edge_index` (`torch.LongTensor`)
            * A `torch.LongTensor` object of shape `(E, 2)`, where `E` is the number of the edges.
        
        @ `position` (`torch.Tensor`)
            * A `torch.Tensor` object of shape `(N, k)`, where `k` is the dimension of the physical domain.
        """
        # Map to the query/key/value spaces (resultant shape: `(N, hidden_channels)`)
        query   = torch.einsum(self.einsum_command, [node_attr, self.sa_query]).reshape(-1, self.channels)
        key     = torch.einsum(self.einsum_command, [node_attr, self.sa_key]).reshape(-1, self.channels)
        value   = torch.einsum(self.einsum_command, [node_attr, self.sa_value]).reshape(-1, self.channels)
        
        # Do message passing and compute the skip connection, if required
        prop    = self.propagate(
            edge_index,
            query = query, key = key, value = value,
            position = position,
        )
        if self.use_linear:
            skip = self.linear.forward(node_attr)
        else:
            skip = 0
            
        # Return the output
        return prop + skip
        

    def message(
            self,
            query_i:    torch.Tensor,
            key_j:      torch.Tensor,
            value_j:    torch.Tensor,
            position_i: torch.Tensor,
            position_j: torch.Tensor,
        ) -> torch.Tensor:
        """
        ### Note
        
        Here the shapes of the input tensors are summarized.
        * Query, key, value: `(N, C)`. Note that the features belonging to distinct heads are combined; otherwise, the message passing algorithm does not work.
        * Position: `(N, k)`
        
        With the above input tensors, the message passing algorithm returns a tensor of shape `(N, C)`.
        Here, `N` is the number of the nodes (points), `H` is the number of the heads, `C` is the number of the channels, and `k` is the dimension of the space in which the points exist.
        """
        qkv_shape = (-1, self.n_heads, self.channels//self.n_heads)
        query_i = query_i.reshape(qkv_shape)
        key_j   = key_j.reshape(qkv_shape)
        value_j = value_j.reshape(qkv_shape)
        
        vector_weights = query_i - key_j + self.pos_encode(position_i - position_j)
        if self.__use_softmax:
            vector_weights = torch.softmax(vector_weights, dim=-1)
        vectors = value_j + self.pos_encode(position_j)
        
        return (vector_weights * vectors).reshape(-1, self.channels)
    
    
    def __repr__(self) -> str:
        channels    = self.channels
        n_heads     = self.n_heads
        use_softmax = self.use_softmax
        use_linear  = self.use_linear
        return f"VectorSelfAttention({channels=}, {n_heads=}, {use_softmax=}, {use_linear=})"


##################################################
##################################################
class HyperLinearSelfAttention(nn.Module):
    """## Linear self-attention with hyper-networks
    
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
        
        return
    
    
    @property
    def einsum_domain(self) -> str:
        return self.__EINSUM_DOMAIN
    @property
    def einsum_command(self) -> str:
        return self.__EINSUM_COMMAND
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
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
##################################################
# End of file