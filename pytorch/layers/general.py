from    typing                  import  *
from    typing_extensions       import  Self

from    math        import  prod

import  torch
from    torch       import  nn

from    ._base_module   import  BaseModule
from    ..utils     import  Objects, EINSUM_STRING, get_activation


##################################################
##################################################
__all__ = [
    "MLP", "HyperMLP",
    "PatchEmbedding",
    "LinearSelfAttention",
    "LinearCrossAttention",
]


##################################################
##################################################
class MLP(nn.Module):
    """## Multi-layer perceptron
    
    -----
    ### Description
    By passing the dimension of the input/output spaces and the hidden spaces, this class constructs a multi-layer perceptron.
    """
    def __init__(
            self,
            channels:           Sequence[int],
            bias:               bool    = True,
            activation_name:    str     = "tanh",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of the class `MLP`
        
        -----
        ### Arguments
        * `channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the output layer.
        * `bias` (`bool`, default: `True`)
            * The `bias` argument of `torch.nn.Linear`.
        * `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        self.__check_channels(channels)
        
        # Save some member variables for representation
        self.__channels         = tuple(channels)
        self.__bias             = bias
        self.__activation_name  = activation_name
        
        # Define the MLP
        self.net = nn.Sequential()
        for cnt in range(len(channels)-2):
            self.net.append(nn.Linear(channels[cnt], channels[cnt + 1], bias = bias))
            self.net.append(get_activation(activation_name, activation_kwargs))
        self.net.append(nn.Linear(channels[-2], channels[-1], bias = bias))
        
        return

    
    @property
    def channels(self) -> tuple[int]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net.forward(X)
    
    
    def __check_channels(self, channels: Sequence[int]) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return
    
    
    def __repr__(self) -> str:
        return f"MLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"


class HyperMLP(nn.Module):
    def __init__(
            self,
            channels:           Sequence[int],
            hyper_channels:     Sequence[int],
            bias:               bool                = True,
            activation_name:    str                 = "tanh",
            activation_kwargs:  dict[str, object]   = {},
        ) -> Self:
        """## The initializer of the class `HyperMLP`
        
        -----
        ### Arguments
        * `channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the output layer.
            
        * `hyper_channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the pre-out layer.
            * *Important* For each hypernet, the last linear layer will be automatically appended using `torch.nn.LazyLinear`.
        
        * `bias` (`bool`, default: `True`)
            * The `bias` argument of `torch.nn.Linear`.
        
        * `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        self.__check_channels(channels)
        
        # Save some member variables for representation
        self.__channels         = tuple(channels)
        self.__bias             = bias
        self.__activation_name      = activation_name
        self.__activation_kwargs    = activation_kwargs
        
        # Define variables for hypernetworks
        __shape_of_weights:    list[tuple[int]]    = []
        __shape_of_biases:     list[tuple[int]]    = []
        for idx in range(len(channels)-1):
            ch_in, ch_out = channels[idx], channels[idx + 1]
            __shape_of_weights.append((ch_out, ch_in))
            __shape_of_biases.append((ch_out,))
        self.__shape_of_weights:    tuple[tuple[int]]    = tuple(__shape_of_weights)
        self.__shape_of_biases:     tuple[tuple[int]]    = tuple(__shape_of_biases)
        
        # Define the hypernetworks
        hypernet_weight = [
            MLP(
                (*hyper_channels, h*w),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
            for (h, w) in self.__shape_of_weights
        ]
        hypernet_bias   = [] if not bias else [
            MLP(
                (*hyper_channels, h),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
            for (h,) in self.__shape_of_biases
        ]
        self.hypernet_weight    = nn.ModuleList(hypernet_weight)
        self.hypernet_bias      = nn.ModuleList(hypernet_bias)
        
        return

    
    @property
    def channels(self) -> tuple[int]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        weights = [net_w.forward(p) for net_w in self.hypernet_weight]
        biases  = [net_b.forward(p) for net_b in self.hypernet_bias] if self.__bias \
            else  [
                torch.zeros(p.size(0), 1, s, dtype=X.dtype)
                for s in self.__shape_of_biases
            ]
        for cnt, w, b in enumerate(zip(weights, biases)):
            w = torch.reshape(w, (-1, *self.__shape_of_weights[cnt]))
            b = torch.reshape(b, (-1, 1, *self.__shape_of_biases[cnt]))
            X = torch.einsum('bi, bji -> bj', X, w) + b
            if cnt < len(weights)-1:
                X = get_activation(self.__activation_name, self.__activation_kwargs)(X)
        return X
        
    
    
    def __check_channels(self, channels: Sequence[int]) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return
    
    
    def __repr__(self) -> str:
        return f"HyperMLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"


##################################################
##################################################
class PatchEmbedding(nn.Module):
    """## Patch embedding
    
    -----
    ### Description
    This class aims to convert the input tensor into a sequence of patches.
    """
    def __init__(
            self,
            dim_domain:         int,
            in_channels:        int,
            patch_size:         Objects[int],
            dim_embed:          Optional[int] = None,
        ) -> Self:
        """## The initializer of the class `PatchEmbedding`
        
        -----
        ### Arguments
        * `dim_domain` (`int`)
            - The dimension of the domain.
        
        * `in_channels` (`int`)
            - The number of the channels in the input tensor.
        
        * `patch_size` (`Objects[int]`)
            - The size of the patch in each dimension.
            
        * `dim_embed` (`Optional[int]`, default: `None`)
            - The number of the channels in the output tensor.
            - If `None`, it is set to the `in_channels * prod(patch_size)`.
        """
        super().__init__()
        
        if dim_domain not in [1, 2, 3]:
            raise ValueError(f"The dimension of the domain should be 1, 2, or 3. ('dim_domain': {dim_domain})")
        if hasattr(patch_size, '__iter__') and len(patch_size) != dim_domain:
            raise ValueError(f"The length of the patch size should be equal to the dimension of the domain. ('dim_domain': {dim_domain}, 'patch_size': {patch_size})")
        
        if not hasattr(patch_size, '__iter__'):
            patch_size = [patch_size for _ in range(dim_domain)]
        
        self.__dim_domain:  int         = dim_domain
        self.__in_channels: int         = in_channels
        self.__patch_size:  Tuple[int]  = tuple(patch_size)
        self.__dim_embed:   int         = in_channels * prod(patch_size) if dim_embed is None else dim_embed
        
        self.patch_embed:   nn.Module = getattr(nn, f"Conv{dim_domain}d")(
            in_channels  = self.__in_channels,
            out_channels = self.__dim_embed,
            kernel_size  = self.__patch_size,
            stride       = self.__patch_size,
        )
        
        return
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def patch_size(self) -> tuple[int]:
        return self.__patch_size
    @property
    def dim_embed(self) -> int:
        return self.__dim_embed
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.permute(self.__permute_1)
        X = self.patch_embed.forward(X)
        X = X.permute(self.__permute_2)
        return X
    
    
    @property
    def __permute_1(self) -> tuple[int]:
        return tuple((0, -1, *tuple(range(1, 1+self.dim_domain))))
    @property
    def __permute_2(self) -> tuple[int]:
        return tuple((0, *tuple(range(2, 2+self.dim_domain)), 1))
    


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
        _size   = (hidden_channels, n_heads, hidden_channels // n_heads)    # (C, H, C/H)
        _scale  = (hidden_channels ** 2) / n_heads
        # Feature maps
        self.sa_query   = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
        self.sa_key     = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
        self.sa_value   = nn.Parameter(torch.randn(size = _size, dtype = torch.float) / _scale)
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
# End of file